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
float minlen = 0.4;         //minima distancia del lidar, aligned with velodyne_nodelet
float max_FOV = 3.0;        //en radianes angulo maximo de vista de la camara
float min_FOV = 0.4;        //en radianes angulo minimo de vista de la camara

///parametros para convertir nube de puntos en imagen
float angular_resolution_x = 0.25f; //from launch file
float angular_resolution_y = 0.46875f; //30° / 64 rows
float max_angle_width = 360.0f;
float max_angle_height = 30.0f;    //match VLP-16 vertical FOV (-15° to +15°)
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth = 100.0;
float min_depth = 8.0;
double max_var = 50.0; 

float interpol_value = 10.0; //from launch file, overridden for dynamic interpolation

bool f_pc = true; 

//input topics 
std::string imgTopic = "/usb_cam/image_raw";
std::string pcTopic = "/velodyne_points";

//matrix calibration lidar and camera
Eigen::MatrixXf Tlc(3,1); //translation matrix lidar-camera
Eigen::MatrixXf Rlc(3,3); //rotation matrix lidar-camera
Eigen::MatrixXf Mc(3,4);  //camera calibration matrix

//range image parametros
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

    //////filter point cloud 
    if (msg_pointCloud == NULL) return;

    PointCloud::Ptr cloud_in (new PointCloud);
    PointCloud::Ptr cloud_out (new PointCloud);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);
    ROS_INFO("Point cloud size after NaN removal: %lu", cloud_in->points.size());
  
    for (int i = 0; i < (int) cloud_in->points.size(); i++)
    {
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);     
        if(distance < minlen || distance > maxlen)
            continue;        
        cloud_out->push_back(cloud_in->points[i]);     
    }  
    ROS_INFO("Point cloud size after distance filter: %lu", cloud_out->points.size());

    //point cloud to image 
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                        pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                        sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;
    ROS_INFO("Range image created: width=%d, height=%d, points=%lu", cols_img, rows_img, cloud_out->points.size());

    arma::mat Z;  //interpolation de la imagen
    arma::mat Zz; //interpolation de las alturas de la imagen
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

    //VLP-16 has 16 layers, vertical FOV from -15 to +15 degrees (30 degrees total)
    const int num_layers = 16;
    const float vlp16_min_angle = -15.0f; // Minimum vertical angle in degrees
    const float vlp16_max_angle = 15.0f;  // Maximum vertical angle in degrees
    const int base_lines = 4; // Starting number of interpolated lines
    const int max_lines = 19; // Maximum number of interpolated lines

    //VLP-16 vertical angles from log (in radians)
    std::vector<float> vlp16_angles = {
        -0.261799, -0.226893, -0.191986, -0.157080, -0.122173, -0.087266, -0.052360, -0.017453,
        0.017453, 0.052360, 0.087266, 0.122173, 0.157080, 0.191986, 0.226893, 0.261799
    };

    arma::vec X = arma::regspace(1, cols_img);  //X = horizontal spacing
    arma::vec XI = arma::regspace(X.min(), 1.0, X.max()); //Same horizontal resolution

    //Pre-calculate total output rows
    int total_output_rows = 0;
    std::vector<int> rows_per_layer(num_layers);
    for (int layer = 0; layer < num_layers; ++layer)
    {
        int num_lines = base_lines + layer; //4 to 19
        rows_per_layer[layer] = num_lines;
        total_output_rows += num_lines; //Sum of 4 + 5 + ... + 19 = 176
    }

    ROS_INFO("Estimated total output rows: %d", total_output_rows);

    arma::mat ZI(total_output_rows, cols_img); //Size based on exact output rows
    arma::mat ZzI(total_output_rows, cols_img);
    ZI.zeros();
    ZzI.zeros();

    //Map angles to row indices
    std::vector<int> row_indices(num_layers + 1);
    for (int i = 0; i < num_layers; ++i)
    {
        float angle_deg = vlp16_angles[i] * 180.0f / M_PI;
        float normalized = (vlp16_max_angle - angle_deg) / (vlp16_max_angle - vlp16_min_angle);
        row_indices[i] = std::round(normalized * (rows_img - 1));
    }
    row_indices[num_layers] = rows_img;

    //Ensure unique and sorted indices
    std::sort(row_indices.begin(), row_indices.end());
    row_indices.erase(std::unique(row_indices.begin(), row_indices.end()), row_indices.end());

    //Fallback to uniform division if insufficient indices
    if (row_indices.size() <= num_layers)
    {
        ROS_WARN("Insufficient unique row indices (%lu), expected %d. Using uniform division.", row_indices.size(), num_layers);
        row_indices.resize(num_layers + 1);
        float rows_per_layer_float = static_cast<float>(rows_img) / num_layers;
        for (int i = 0; i <= num_layers; ++i)
        {
            row_indices[i] = std::round(i * rows_per_layer_float);
        }
        row_indices[num_layers] = rows_img;
    }

    ROS_INFO("Row indices: ");
    for (size_t i = 0; i < row_indices.size(); ++i)
    {
        ROS_INFO("  [%lu] = %d", i, row_indices[i]);
    }

    //Interpolate each layer
    int current_row_output = 0;
    for (int layer = 0; layer < num_layers; ++layer)
    {
        int num_lines = base_lines + layer; //4 to 19
        int row_start = row_indices[layer];
        int row_end = row_indices[layer + 1];
        if (row_end <= row_start + 1)
        {
            row_end = row_start + 2;
            if (row_end > rows_img) row_end = rows_img;
        }

        arma::mat Z_layer = Z.rows(row_start, row_end - 1);
        arma::mat Zz_layer = Zz.rows(row_start, row_end - 1);
        arma::vec Y_layer = arma::regspace(row_start + 1, row_end);

        if (Y_layer.n_elem < 2 || Z_layer.n_rows < 2)
        {
            ROS_WARN("Skipping layer %d: insufficient rows (start=%d, end=%d, Y_layer.size=%lu, Z_layer.rows=%lu)",
                     layer, row_start, row_end - 1, Y_layer.n_elem, Z_layer.n_rows);
            continue;
        }

        arma::vec YI_layer = arma::regspace(row_start + 1, 1.0, row_start + num_lines); //Exactly num_lines rows
        ROS_INFO("Layer %d: rows %d to %d, num_lines=%d, Y_layer.size=%lu, YI_layer.size=%lu",
                 layer, row_start, row_end - 1, num_lines, Y_layer.n_elem, YI_layer.n_elem);

        arma::mat ZI_layer, ZzI_layer;
        try
        {
            arma::interp2(X, Y_layer, Z_layer, XI, YI_layer, ZI_layer, "linear");
            arma::interp2(X, Y_layer, Zz_layer, XI, YI_layer, ZzI_layer, "linear");
        }
        catch (const std::exception& e)
        {
            ROS_ERROR("Interpolation error in layer %d: %s", layer, e.what());
            continue;
        }

        //Copy to output matrices
        for (uint i = 0; i < ZI_layer.n_rows && current_row_output + i < total_output_rows; ++i)
        {
            for (uint j = 0; j < ZI_layer.n_cols && j < cols_img; ++j)
            {
                ZI(current_row_output + i, j) = ZI_layer(i, j);
                ZzI(current_row_output + i, j) = ZzI_layer(i, j);
            }
        }
        current_row_output += ZI_layer.n_rows;
    }

    if (current_row_output == 0)
    {
        ROS_ERROR("No valid interpolated rows produced. Check range image configuration.");
        return;
    }

    //Resize to actual output size
    ZI = ZI.rows(0, current_row_output - 1);
    ZzI = ZzI.rows(0, current_row_output - 1);
    ROS_INFO("Interpolated output: ZI.rows=%lu, ZI.cols=%lu", ZI.n_rows, ZI.n_cols);

    //Handle zeros in interpolation
    arma::mat Zout = ZI;
    for (uint i = 0; i < ZI.n_rows; i++)
    {
        for (uint j = 0; j < ZI.n_cols; j++)
        {
            if (ZI(i, j) == 0)
            {
                if (i + num_layers < ZI.n_rows)
                {
                    for (int k = 1; k <= num_layers; k++)
                    {
                        Zout(i + k, j) = 0;
                    }
                }
                if (i > num_layers)
                {
                    for (int k = 1; k <= num_layers; k++)
                    {
                        Zout(i - k, j) = 0;
                    }
                }
            }
        }
    }
    ZI = Zout;

    //Determine data density threshold
    double density_threshold = 0.05;
    int max_window_size = 9;
    int min_window_size = 3;

    for (uint i = 1; i < ZI.n_rows - 1; ++i)
    {
        for (uint j = 1; j < ZI.n_cols - 1; ++j)
        {
            if (ZI(i, j) == 0)
            {
                double weighted_sum = 0.0;
                double weight_total = 0.0;
                int valid_neighbors = 0;
                for (int di = -1; di <= 1; ++di)
                {
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0)
                        {
                            valid_neighbors++;
                        }
                    }
                }
                int window_size = (valid_neighbors < density_threshold * 9) ? max_window_size : min_window_size;
                for (int di = -window_size; di <= window_size; ++di)
                {
                    for (int dj = -window_size; dj <= window_size; ++dj)
                    {
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0)
                        {
                            double distance = std::sqrt(di * di + dj * dj);
                            double weight = 1.0 / (distance + 1e-6);
                            weighted_sum += ZI(ni, nj) * weight;
                            weight_total += weight;
                        }
                    }
                }
                if (weight_total > 0)
                {
                    ZI(i, j) = weighted_sum / weight_total;
                }
            }
        }
    }

    //Compute gradients for edge detection
    arma::mat Zenhanced = ZI;
    arma::mat grad_x = arma::zeros(ZI.n_rows, ZI.n_cols);
    arma::mat grad_y = arma::zeros(ZI.n_rows, ZI.n_cols);
    arma::mat grad_mag = arma::zeros(ZI.n_rows, ZI.n_cols);

    for (uint i = 1; i < ZI.n_rows - 1; ++i)
    {
        for (uint j = 1; j < ZI.n_cols - 1; ++j)
        {
            if (ZI(i, j) > 0)
            {
                grad_x(i, j) = (ZI(i, j + 1) - ZI(i, j - 1)) * 0.5;
                grad_y(i, j) = (ZI(i + 1, j) - ZI(i - 1, j)) * 0.5;
                grad_mag(i, j) = std::sqrt(grad_x(i, j) * grad_x(i, j) + grad_y(i, j) * grad_y(i, j));
            }
        }
    }

    double edge_threshold = 0.1 * arma::max(arma::max(grad_mag));
    for (uint i = 1; i < ZI.n_rows - 1; ++i)
    {
        for (uint j = 1; j < ZI.n_cols - 1; ++j)
        {
            if (grad_mag(i, j) > edge_threshold)
            {
                double weight = std::max(0.0, 1.0 - grad_mag(i, j) / edge_threshold);
                Zenhanced(i, j) = ZI(i, j) * weight + Zenhanced(i, j) * (1 - weight);
            }
        }
    }
    ZI = Zenhanced;

    if (f_pc)
    {    
        for (uint i = 0; i < ((ZI.n_rows-1)/num_layers); i += 1)       
        {
            for (uint j = 0; j < ZI.n_cols-5 ; j += 1)
            {
                double promedio = 0;
                double varianza = 0;
                int valid_count = 0;
                for (uint k = 0; k < num_layers ; k += 1)
                {
                    if ((i * num_layers) + k < ZI.n_rows)
                    {
                        promedio += ZI((i * num_layers) + k, j);
                        valid_count++;
                    }
                }
                if (valid_count > 0)
                {
                    promedio /= valid_count;    
                    for (uint l = 0; l < num_layers; l++) 
                    {
                        if ((i * num_layers) + l < ZI.n_rows)
                        {
                            varianza += pow((ZI((i * num_layers) + l, j) - promedio), 2.0);  
                        }
                    }
                    if (varianza > max_var)
                    {
                        for (uint m = 0; m < num_layers; m++) 
                        {
                            if ((i * num_layers) + m < ZI.n_rows)
                            {
                                Zout((i * num_layers) + m, j) = 0;                 
                            }
                        }
                    }
                }
            }
        }
        ZI = Zout;
    }

    //imagen de rango a nube de puntos  
    int num_pc = 0; 
    PointCloud::Ptr point_cloud (new PointCloud);
    PointCloud::Ptr cloud (new PointCloud);
    point_cloud->width = ZI.n_cols; 
    point_cloud->height = ZI.n_rows;
    point_cloud->is_dense = false;
    point_cloud->points.resize(point_cloud->width * point_cloud->height);

    for (uint i = 0; i < ZI.n_rows - num_layers; i += 1)
    {       
        for (uint j = 0; j < ZI.n_cols ; j += 1)
        {
            float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));
            if (ang < min_FOV - M_PI/2.0 || ang > max_FOV - M_PI/2.0) 
                continue;

            if (!(Zout(i,j) == 0))
            {  
                float pc_modulo = Zout(i,j);
                float pc_x = sqrt(pow(pc_modulo,2) - pow(ZzI(i,j),2)) * cos(ang);
                float pc_y = sqrt(pow(pc_modulo,2) - pow(ZzI(i,j),2)) * sin(ang);

                float ang_x_lidar = 0.6 * M_PI / 180.0;  
                Eigen::MatrixXf Lidar_matrix(3,3);
                Eigen::MatrixXf result(3,1);
                Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                                0, 1, 0,
                                -sin(ang_x_lidar), 0, cos(ang_x_lidar);
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

    PointCloud::Ptr P_out (new PointCloud);
    P_out = cloud;

    Eigen::MatrixXf RTlc(4,4);
    RTlc << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
            Rlc(1), Rlc(4), Rlc(7), Tlc(1),
            Rlc(2), Rlc(5), Rlc(8), Tlc(2),
            0, 0, 0, 1;

    int size_inter_Lidar = (int) P_out->points.size();
    Eigen::MatrixXf Lidar_camera(3, size_inter_Lidar);
    Eigen::MatrixXf Lidar_cam(3,1);
    Eigen::MatrixXf pc_matrix(4,1);
    Eigen::MatrixXf pointCloud_matrix(4, size_inter_Lidar);

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
        px_data = (int)(Lidar_cam(0,0) / Lidar_cam(2,0));
        py_data = (int)(Lidar_cam(1,0) / Lidar_cam(2,0));
      
        if (px_data < 0.0 || px_data >= cols || py_data < 0.0 || py_data >= rows)
            continue;

        int color_dis_x = (int)(255 * ((P_out->points[i].x) / maxlen));
        int color_dis_z = (int)(255 * ((P_out->points[i].x) / 10.0));
        if (color_dis_z > 255)
            color_dis_z = 255;

        cv::Vec3b & color = color_pcl->image.at<cv::Vec3b>(py_data, px_data);
        point.x = P_out->points[i].x;
        point.y = P_out->points[i].y;
        point.z = P_out->points[i].z;
        point.r = (int)color[2]; 
        point.g = (int)color[1]; 
        point.b = (int)color[0];
        pc_color->points.push_back(point);   
        cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1, CV_RGB(255-color_dis_x, (int)(color_dis_z), color_dis_x), cv::FILLED);
    }
    pc_color->is_dense = true;
    pc_color->width = (int) pc_color->points.size();
    pc_color->height = 1;
    pc_color->header.frame_id = "velodyne";

    pcOnimg_pub.publish(cv_ptr->toImageMsg());
    pc_pub.publish(pc_color);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointCloudOntImage");
    ros::NodeHandle nh;  

    ///Load Parameters
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
