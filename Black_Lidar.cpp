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

// Parameters
float maxlen = 100.0;       // Maximum distance
float minlen = 0.01;        // Minimum distance
float max_FOV = 3.0;        // Max FOV in radians
float min_FOV = 0.4;        // Min FOV in radians

// Conversion parameters
float angular_resolution_x = 0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth = 100.0;
float min_depth = 8.0;
double max_var = 50.0; 

float base_interpol_value = 20.0f;
float max_interpol_value = 30.0f;

bool f_pc = true; 

// Topics
std::string imgTopic = "/camera/color/image_raw";
std::string pcTopic = "/velodyne_points";

// Calibration matrices
Eigen::MatrixXf Tlc(3,1); // Translation matrix
Eigen::MatrixXf Rlc(3,3); // Rotation matrix
Eigen::MatrixXf Mc(3,4);  // Camera matrix

// Range image parameters
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

/////////////////////////////////////// Dynamic Interpolation Functions

float getDynamicInterpolationFactor(float distance, float max_distance) {
    // Linear interpolation between base and max values
    float ratio = distance / max_distance;
    ratio = min(max(ratio, 0.0f), 1.0f); // Clamp to [0,1]
    return base_interpol_value + ratio * (max_interpol_value - base_interpol_value);
}

void applyDynamicInterpolation(arma::mat& ZI, const arma::vec& X, const arma::vec& Y, float max_distance) {
    for (uint j = 0; j < ZI.n_cols; j++) {
        vector<pair<uint, float>> valid_points;
        
        // Find all valid points in column
        for (uint i = 0; i < ZI.n_rows; i++) {
            if (ZI(i,j) > 0) {
                valid_points.push_back({i, ZI(i,j)});
            }
        }
        
        // Sort by row index
        sort(valid_points.begin(), valid_points.end());
        
        // Interpolate between points
        for (size_t k = 0; k + 1 < valid_points.size(); k++) {
            uint start_row = valid_points[k].first;
            uint end_row = valid_points[k+1].first;
            float start_val = valid_points[k].second;
            float end_val = valid_points[k+1].second;
            
            float avg_dist = (start_val + end_val) / 2.0f;
            float interpol_factor = getDynamicInterpolationFactor(avg_dist, max_distance);
            
            uint steps = end_row - start_row;
            if (steps <= 1) continue;
            
            uint interpol_points = min(static_cast<uint>(interpol_factor), steps - 1);
            if (interpol_points == 0) continue;
            
            for (uint p = 1; p <= interpol_points; p++) {
                float t = static_cast<float>(p) / (interpol_points + 1);
                uint target_row = start_row + static_cast<uint>(t * steps);
                
                if (target_row < ZI.n_rows && ZI(target_row,j) == 0) {
                    ZI(target_row,j) = start_val + t * (end_val - start_val);
                }
            }
        }
    }
}

/////////////////////////////////////// Callback Function

void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2, const ImageConstPtr& in_image)
{
    // Convert ROS image to OpenCV
    cv_bridge::CvImagePtr cv_ptr, color_pcl;
    try {
        cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
        color_pcl = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Convert point cloud
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*in_pc2,pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2,*msg_pointCloud);
    
    if (msg_pointCloud == NULL) return;

    // Filter point cloud
    PointCloud::Ptr cloud_in (new PointCloud);
    PointCloud::Ptr cloud_out (new PointCloud);
    std::vector<int> indices;
    
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);
    
    for (int i = 0; i < (int)cloud_in->points.size(); i++) {
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + 
                             cloud_in->points[i].y * cloud_in->points[i].y);     
        if(distance >= minlen && distance <= maxlen) {
            cloud_out->push_back(cloud_in->points[i]);
        }
    }

    // Create range image
    Eigen::Affine3f sensorPose = Eigen::Affine3f::Identity();
    rangeImage->createFromPointCloud(*cloud_out, 
                                   pcl::deg2rad(angular_resolution_x), 
                                   pcl::deg2rad(angular_resolution_y),
                                   pcl::deg2rad(max_angle_width), 
                                   pcl::deg2rad(max_angle_height),
                                   sensorPose, 
                                   coordinate_frame);

    // Prepare matrices
    arma::mat Z(rangeImage->height, rangeImage->width, arma::fill::zeros);
    arma::mat Zz(rangeImage->height, rangeImage->width, arma::fill::zeros);

    for (int i = 0; i < rangeImage->width; ++i) {
        for (int j = 0; j < rangeImage->height; ++j) {
            float r = rangeImage->getPoint(i, j).range;     
            float zz = rangeImage->getPoint(i, j).z; 
       
            if(!std::isinf(r) && r >= minlen && r <= maxlen && !std::isnan(zz)) {
                Z(j,i) = r;   
                Zz(j,i) = zz;
            }
        }
    }

    // Interpolate
    arma::vec X = arma::regspace(1, Z.n_cols);
    arma::vec Y = arma::regspace(1, Z.n_rows);
    arma::vec XI = arma::regspace(X.min(), X.max());
    arma::vec YI = arma::regspace(Y.min(), 1.0/base_interpol_value, Y.max());

    arma::mat ZI, ZzI;
    arma::interp2(X, Y, Z, XI, YI, ZI, "linear");
    arma::interp2(X, Y, Zz, XI, YI, ZzI, "linear");

    // Apply dynamic interpolation
    applyDynamicInterpolation(ZI, X, Y, maxlen);

    // Filter interpolated points
    arma::mat Zout = ZI;
    for (uint i = 0; i < ZI.n_rows; i++) {
        for (uint j = 0; j < ZI.n_cols; j++) {
            if (ZI(i,j) == 0 && i + base_interpol_value < ZI.n_rows) {
                Zout.rows(i, i+base_interpol_value-1).col(j).zeros();
            }
        }
    }
    ZI = Zout;

    // Convert back to point cloud
    PointCloud::Ptr point_cloud(new PointCloud);
    PointCloud::Ptr cloud(new PointCloud);
    
    point_cloud->width = ZI.n_cols; 
    point_cloud->height = ZI.n_rows;
    point_cloud->is_dense = false;
    point_cloud->points.resize(point_cloud->width * point_cloud->height);

    int num_pc = 0;
    for (uint i = 0; i < ZI.n_rows - base_interpol_value; i += 1) {
        for (uint j = 0; j < ZI.n_cols; j += 1) {
            float ang = M_PI-((2.0 * M_PI * j)/(ZI.n_cols));

            if (ang >= (min_FOV-M_PI/2.0) && ang <= (max_FOV-M_PI/2.0) && Zout(i,j) != 0) {  
                float pc_modulo = Zout(i,j);
                float pc_x = sqrt(pow(pc_modulo,2)-pow(ZzI(i,j),2)) * cos(ang);
                float pc_y = sqrt(pow(pc_modulo,2)-pow(ZzI(i,j),2)) * sin(ang);

                Eigen::MatrixXf Lidar_matrix(3,3);
                Eigen::MatrixXf result(3,1);
                
                Lidar_matrix << cos(0.6*M_PI/180.0), 0, sin(0.6*M_PI/180.0),
                               0, 1, 0,
                               -sin(0.6*M_PI/180.0), 0, cos(0.6*M_PI/180.0);

                result << pc_x, pc_y, ZzI(i,j);
                result = Lidar_matrix * result;

                point_cloud->points[num_pc].x = result(0);
                point_cloud->points[num_pc].y = result(1);
                point_cloud->points[num_pc].z = result(2);

                cloud->push_back(point_cloud->points[num_pc]); 
                num_pc++;
            }
        }
    }

    // Project to camera
    Eigen::MatrixXf RTlc(4,4);
    RTlc << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
            Rlc(1), Rlc(4), Rlc(7), Tlc(1),
            Rlc(2), Rlc(5), Rlc(8), Tlc(2),
            0, 0, 0, 1;

    unsigned int cols = in_image->width;
    unsigned int rows = in_image->height;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (size_t i = 0; i < cloud->size(); i++) {
        Eigen::MatrixXf pc_matrix(4,1);
        pc_matrix << -cloud->points[i].y,   
                     -cloud->points[i].z,   
                     cloud->points[i].x,  
                     1.0;

        Eigen::MatrixXf Lidar_cam = Mc * (RTlc * pc_matrix);

        int px_data = (int)(Lidar_cam(0,0)/Lidar_cam(2,0));
        int py_data = (int)(Lidar_cam(1,0)/Lidar_cam(2,0));
        
        if(px_data < 0 || px_data >= (int)cols || py_data < 0 || py_data >= (int)rows)
            continue;

        // Color point cloud
        cv::Vec3b color = color_pcl->image.at<cv::Vec3b>(py_data, px_data);
        pcl::PointXYZRGB point;
        point.x = cloud->points[i].x;
        point.y = cloud->points[i].y;
        point.z = cloud->points[i].z;
        point.r = color[2];
        point.g = color[1];
        point.b = color[0];
        
        pc_color->points.push_back(point);
        cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1, 
                  cv::Scalar(255-(int)(255*(cloud->points[i].x/maxlen)),
                            (int)(255*(cloud->points[i].x/10.0)),
                            (int)(255*(cloud->points[i].x/maxlen))),
                  cv::FILLED);
    }

    // Publish results
    pc_color->is_dense = true;
    pc_color->width = pc_color->size();
    pc_color->height = 1;
    pc_color->header.frame_id = "velodyne";

    pcOnimg_pub.publish(cv_ptr->toImageMsg());
    pc_pub.publish(pc_color);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointCloudOnImage");
    ros::NodeHandle nh;  
    
    // Load parameters
    nh.getParam("/maxlen", maxlen);
    nh.getParam("/minlen", minlen);
    nh.getParam("/max_ang_FOV", max_FOV);
    nh.getParam("/min_ang_FOV", min_FOV);
    nh.getParam("/pcTopic", pcTopic);
    nh.getParam("/imgTopic", imgTopic);
    nh.getParam("/max_var", max_var);  
    nh.getParam("/filter_output_pc", f_pc);
    nh.getParam("/x_resolution", angular_resolution_x);
    nh.getParam("/base_interpolation", base_interpol_value);
    nh.getParam("/max_interpolation", max_interpol_value);
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

    // Setup subscribers
    message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic, 1);
    message_filters::Subscriber<Image> img_sub(nh, imgTopic, 1);

    typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));
    
    // Setup publishers
    pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);
    rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
    pc_pub = nh.advertise<PointCloud>("/points2", 1);  

    ros::spin();
}
