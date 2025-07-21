#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_transport/image_transport.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <math.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <armadillo>

#include <chrono> 

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

//Publisher
ros::Publisher pcOnimg_pub;
ros::Publisher pc_pub;

float maxlen = 100.0;       //maxima distancia del lidar
float minlen = 0.0;        //minima distancia del lidar (updated to match launch file)
float max_FOV = 2.7;       // en radianes angulo maximo de vista de la camara (updated)
float min_FOV = 0.5;       // en radianes angulo minimo de vista de la camara (updated)

/// parametros para convertir nube de puntos en imagen
float angular_resolution_x = 0.25f; // Updated to match launch file
float angular_resolution_y = 0.5f;  // Finer resolution for more rows
float max_angle_width = 360.0f;
float max_angle_height = 30.0f;    // Match VLP-16 vertical FOV (-15° to +15°)
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth = 100.0;
float min_depth = 8.0;
double max_var = 50.0; 

float interpol_value = 10.0; // Base interpolation value, overridden for dynamic interpolation

bool f_pc = true; 

// input topics 
std::string imgTopic = "/usb_cam/image_raw"; // Updated to match launch file
std::string pcTopic = "/velodyne_points";

//matrix calibration lidar and camera
Eigen::MatrixXf Tlc(3,1); // translation matrix lidar-camera
Eigen::MatrixXf Rlc(3,3); // rotation matrix lidar-camera
Eigen::MatrixXf Mc(3,4);  // camera calibration matrix

// range image parametros
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

///////////////////////////////////////callback
void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2 , const ImageConstPtr& in_image)
{
    cv_bridge::CvImagePtr cv_ptr , color_pcl;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
        color_pcl = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    //Conversion from sensor_msgs::PointCloud2 to pcl::PointCloud<T>
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*in_pc2,pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2,*msg_pointCloud);

    ////// filter point cloud 
    if (msg_pointCloud == NULL) return;

    PointCloud::Ptr cloud_in (new PointCloud);
    PointCloud::Ptr cloud_out (new PointCloud);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);
  
    ROS_INFO("Point cloud size after NaN removal: %zu", cloud_in->points.size());
    for (int i = 0; i < (int) cloud_in->points.size(); i++)
    {
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);     
        if(distance < minlen || distance > maxlen)
            continue;        
        cloud_out->push_back(cloud_in->points[i]);     
    }  
    ROS_INFO("Point cloud size after distance filter: %zu", cloud_out->points.size());

    // point cloud to image 
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                        pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                        sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;
    ROS_INFO("Range image created: width=%d, height=%d", cols_img, rows_img);

    arma::mat Z;  // interpolation de la imagen
    arma::mat Zz; // interpolation de las alturas de la imagen
    Z.zeros(rows_img, cols_img);         
    Zz.zeros(rows_img, cols_img);       

    Eigen::MatrixXf ZZei (rows_img, cols_img);
 
    for (int i = 0; i < cols_img; ++i)
    {
        for (int j = 0; j < rows_img ; ++j)
        {
            float r = rangeImage->getPoint(i, j).range;     
            float zz = rangeImage->getPoint(i, j).z; 
            if(std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz))
            {
                continue;
            }             
            Z.at(j,i) = r;   
            Zz.at(j,i) = zz;
        }
    }

    ////////////////////////////////////////////// FIXED dynamic interpolation
    // VLP-16 has 16 layers, vertical FOV from -15 to +15 degrees (30 degrees total)
    const int num_layers = 8;  // Reduced to work better with available rows
    const float vlp16_min_angle = -15.0f; // Minimum vertical angle in degrees
    const float vlp16_max_angle = 15.0f;  // Maximum vertical angle in degrees
    
    // Calculate interpolation factor based on available rows
    int target_rows = std::max(32, rows_img * 3); // Target at least 32 rows or 3x input
    double interpolation_factor = static_cast<double>(target_rows) / rows_img;
    
    arma::vec X = arma::regspace(0, cols_img - 1);  // 0-based indexing
    arma::vec XI = arma::regspace(0, 1.0, cols_img - 1); // Same horizontal resolution
    arma::vec Y = arma::regspace(0, rows_img - 1);  // Original row indices
    
    // Create interpolated Y coordinates with higher density
    double step = 1.0 / interpolation_factor;
    arma::vec YI = arma::regspace(0, step, rows_img - 1);
    
    ROS_INFO("Interpolation setup: original rows=%d, target rows=%zu, step=%.3f", 
             rows_img, static_cast<size_t>(YI.n_elem), step);
    
    // Perform 2D interpolation
    arma::mat ZI, ZzI;
    try 
    {
        // Check if we have sufficient data for interpolation
        if (Y.n_elem >= 2 && X.n_elem >= 2)
        {
            arma::interp2(X, Y, Z, XI, YI, ZI, "linear", 0.0);  // Use 0.0 for extrapolation
            arma::interp2(X, Y, Zz, XI, YI, ZzI, "linear", 0.0);
            ROS_INFO("Interpolation successful: ZI size = %zu x %zu", 
                     static_cast<size_t>(ZI.n_rows), static_cast<size_t>(ZI.n_cols));
        }
        else
        {
            ROS_WARN("Insufficient data for interpolation, using original data");
            ZI = Z;
            ZzI = Zz;
        }
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("Global interpolation failed: %s", e.what());
        ROS_WARN("Using original range image data");
        ZI = Z;
        ZzI = Zz;
    }

    // Handle zeros in interpolation with improved method
    arma::mat Zout = ZI;
    for (arma::uword i = 1; i < ZI.n_rows - 1; ++i)
    {
        for (arma::uword j = 1; j < ZI.n_cols - 1; ++j)
        {
            if (ZI(i, j) == 0)
            {
                // Simple averaging from valid neighbors
                double sum = 0.0;
                int count = 0;
                for (int di = -1; di <= 1; ++di)
                {
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        if (di == 0 && dj == 0) continue;
                        arma::uword ni = i + di;
                        arma::uword nj = j + dj;
                        if (ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0)
                        {
                            sum += ZI(ni, nj);
                            count++;
                        }
                    }
                }
                if (count >= 3) // Only fill if we have enough neighbors
                {
                    Zout(i, j) = sum / count;
                }
            }
        }
    }
    ZI = Zout;

    // Variance-based filtering (simplified)
    if (f_pc && ZI.n_rows > 4)
    {
        int layer_height = std::max(2, static_cast<int>(ZI.n_rows) / num_layers);
        
        for (arma::uword i = 0; i < ZI.n_rows - static_cast<arma::uword>(layer_height); i += layer_height)
        {
            for (arma::uword j = 0; j < ZI.n_cols; ++j)
            {
                double sum = 0.0;
                int valid_count = 0;
                
                // Calculate mean
                for (int k = 0; k < layer_height && (i + k) < ZI.n_rows; ++k)
                {
                    if (ZI(i + k, j) > 0)
                    {
                        sum += ZI(i + k, j);
                        valid_count++;
                    }
                }
                
                if (valid_count < 2) continue;
                double mean = sum / valid_count;
                
                // Calculate variance
                double variance = 0.0;
                for (int k = 0; k < layer_height && (i + k) < ZI.n_rows; ++k)
                {
                    if (ZI(i + k, j) > 0)
                    {
                        variance += std::pow(ZI(i + k, j) - mean, 2.0);
                    }
                }
                variance /= valid_count;
                
                // Filter high variance regions
                if (variance > max_var)
                {
                    for (int k = 0; k < layer_height && (i + k) < ZI.n_rows; ++k)
                    {
                        ZI(i + k, j) = 0;
                    }
                }
            }
        }
    }

    // Convert range image back to point cloud
    int num_pc = 0; 
    PointCloud::Ptr point_cloud (new PointCloud);
    PointCloud::Ptr cloud (new PointCloud);
    point_cloud->width = ZI.n_cols; 
    point_cloud->height = ZI.n_rows;
    point_cloud->is_dense = false;
    point_cloud->points.resize(point_cloud->width * point_cloud->height);

    for (arma::uword i = 0; i < ZI.n_rows; i += 1)
    {       
        for (arma::uword j = 0; j < ZI.n_cols ; j += 1)
        {
            if (ZI(i,j) <= 0) continue;
            
            float ang = M_PI - ((2.0 * M_PI * static_cast<float>(j)) / (static_cast<float>(ZI.n_cols)));
            if (ang < min_FOV - M_PI/2.0 || ang > max_FOV - M_PI/2.0) 
                continue;

            float pc_modulo = ZI(i,j);
            float z_val = (i < ZzI.n_rows && j < ZzI.n_cols) ? ZzI(i,j) : 0.0f;
            
            // Ensure we don't have invalid sqrt values
            float xy_dist_sq = std::max(0.0f, pc_modulo*pc_modulo - z_val*z_val);
            float xy_dist = std::sqrt(xy_dist_sq);
            
            float pc_x = xy_dist * cos(ang);
            float pc_y = xy_dist * sin(ang);

            // Apply lidar rotation correction
            float ang_x_lidar = 0.6 * M_PI / 180.0;  
            Eigen::MatrixXf Lidar_matrix(3,3);
            Eigen::MatrixXf result(3,1);
            Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                            0, 1, 0,
                            -sin(ang_x_lidar), 0, cos(ang_x_lidar);
            result << pc_x, pc_y, z_val;
            result = Lidar_matrix * result;

            point_cloud->points[num_pc].x = result(0);
            point_cloud->points[num_pc].y = result(1);
            point_cloud->points[num_pc].z = result(2);
            cloud->push_back(point_cloud->points[num_pc]); 
            num_pc++;
        }
    }  

    ROS_INFO("Generated %d points from interpolated range image", num_pc);

    PointCloud::Ptr P_out (new PointCloud);
    P_out = cloud;

    // Camera projection and visualization
    Eigen::MatrixXf RTlc(4,4);
    RTlc << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
            Rlc(1), Rlc(4), Rlc(7), Tlc(1),
            Rlc(2), Rlc(5), Rlc(8), Tlc(2),
            0, 0, 0, 1;

    int size_inter_Lidar = (int) P_out->points.size();
    Eigen::MatrixXf Lidar_cam(3,1);
    Eigen::MatrixXf pc_matrix(4,1);

    unsigned int cols = in_image->width;
    unsigned int rows = in_image->height;

    uint px_data = 0;
    uint py_data = 0;

    pcl::PointXYZRGB point;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color (new pcl::PointCloud<pcl::PointXYZRGB>);

    for (int i = 0; i < size_inter_Lidar; i++)
    {
        pc_matrix(0,0) = -P_out->points[i].y;   
        pc_matrix(1,0) = -P_out->points[i].z;   
        pc_matrix(2,0) = P_out->points[i].x;  
        pc_matrix(3,0) = 1.0;

        Lidar_cam = Mc * (RTlc * pc_matrix);
        
        // Check for valid depth
        if (Lidar_cam(2,0) <= 0) continue;
        
        px_data = (int)(Lidar_cam(0,0) / Lidar_cam(2,0));
        py_data = (int)(Lidar_cam(1,0) / Lidar_cam(2,0));
      
        if (px_data < 0 || px_data >= cols || py_data < 0 || py_data >= rows)
            continue;

        int color_dis_x = (int)(255 * std::min(1.0f, (P_out->points[i].x) / maxlen));
        int color_dis_z = (int)(255 * std::min(1.0f, (P_out->points[i].x) / 10.0f));

        cv::Vec3b & color = color_pcl->image.at<cv::Vec3b>(py_data, px_data);
        point.x = P_out->points[i].x;
        point.y = P_out->points[i].y;
        point.z = P_out->points[i].z;
        point.r = (int)color[2]; 
        point.g = (int)color[1]; 
        point.b = (int)color[0];
        pc_color->points.push_back(point);   
        cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1, 
                  CV_RGB(255-color_dis_x, color_dis_z, color_dis_x), cv::FILLED);
    }
    
    pc_color->is_dense = true;
    pc_color->width = (int) pc_color->points.size();
    pc_color->height = 1;
    pc_color->header.frame_id = "velodyne";
    pcl_conversions::toPCL(in_pc2->header.stamp, pc_color->header.stamp);

    pcOnimg_pub.publish(cv_ptr->toImageMsg());
    pc_pub.publish(pc_color);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pontCloudOntImage");
    ros::NodeHandle nh;  

    /// Load Parameters
    nh.getParam("/maxlen", maxlen);
    nh.getParam("/minlen", minlen);
    nh.getParam("/max_ang_FOV", max_FOV);
    nh.getParam("/min_ang_FOV", min_FOV);
    nh.getParam("/pcTopic", pcTopic);
    nh.getParam("/imgTopic", imgTopic);
    nh.getParam("/max_var", max_var);  
    nh.getParam("/filter_output_pc", f_pc);
    nh.getParam("/x_resolution", angular_resolution_x);
    nh.getParam("/y_interpolation", interpol_value);
    nh.getParam("/ang_Y_resolution", angular_resolution_y);
  
    XmlRpc::XmlRpcValue param;
    nh.getParam("/matrix_file/tlc", param);
    Tlc << (double)param[0], (double)param[1], (double)param[2];
    nh.getParam("/matrix_file/rlc", param);
    Rlc << (double)param[0], (double)param[1], (double)param[2],
           (double)param[3], (double)param[4], (double)param[5],
           (double)param[6], (double)param[7], (double)param[8];
    nh.getParam("/matrix_file/camera_matrix", param);
    Mc << (double)param[0], (double)param[1], (double)param[2], (double)param[3],
          (double)param[4], (double)param[5], (double)param[6], (double)param[7],
          (double)param[8], (double)param[9], (double)param[10], (double)param[11];
    
    message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic , 1);
    message_filters::Subscriber<Image> img_sub(nh, imgTopic, 1);
    typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));
    pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);
    rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
    pc_pub = nh.advertise<PointCloud>("/points2", 1);  

    ros::spin();
}
