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
float minlen = 0.01;        //minima distancia del lidar
float max_FOV = 3.0;        // en radianes angulo maximo de vista de la camara
float min_FOV = 0.4;        // en radianes angulo minimo de vista de la camara

/// parametros para convertir nube de puntos en imagen
float angular_resolution_x = 0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth = 100.0;
float min_depth = 8.0;
double max_var = 50.0; 

// VLP-16 specific parameters
int vlp16_layers = 16;
int base_interpolation = 7;
int interpolation_increment = 2;

bool f_pc = true; 

// input topics 
std::string imgTopic = "/camera/color/image_raw";
std::string pcTopic = "/velodyne_points";

//matrix calibration lidar and camera
Eigen::MatrixXf Tlc(3,1); // translation matrix lidar-camera
Eigen::MatrixXf Rlc(3,3); // rotation matrix lidar-camera
Eigen::MatrixXf Mc(3,4);  // camera calibration matrix

// range image parametros
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

// Function to calculate interpolation value for each layer
int getInterpolationForLayer(int layer) {
    return base_interpolation + (layer * interpolation_increment);
}

// Enhanced edge preservation with adaptive filtering
arma::mat enhancedEdgePreservation(const arma::mat& ZI, const arma::mat& ZzI, int layer_interpol) {
    arma::mat Zenhanced = ZI;
    arma::mat grad_x = arma::zeros(ZI.n_rows, ZI.n_cols);
    arma::mat grad_y = arma::zeros(ZI.n_rows, ZI.n_cols);
    arma::mat grad_mag = arma::zeros(ZI.n_rows, ZI.n_cols);
    arma::mat local_variance = arma::zeros(ZI.n_rows, ZI.n_cols);

    // Calculate gradients and local variance
    for (uint i = 2; i < ZI.n_rows - 2; ++i) {
        for (uint j = 2; j < ZI.n_cols - 2; ++j) {
            if (ZI(i, j) > 0) {
                // Sobel-like gradient calculation for better edge detection
                grad_x(i, j) = (-ZI(i-1, j-1) + ZI(i-1, j+1) - 2*ZI(i, j-1) + 2*ZI(i, j+1) - ZI(i+1, j-1) + ZI(i+1, j+1)) / 8.0;
                grad_y(i, j) = (-ZI(i-1, j-1) - 2*ZI(i-1, j) - ZI(i-1, j+1) + ZI(i+1, j-1) + 2*ZI(i+1, j) + ZI(i+1, j+1)) / 8.0;
                grad_mag(i, j) = std::sqrt(grad_x(i, j) * grad_x(i, j) + grad_y(i, j) * grad_y(i, j));
                
                // Calculate local variance in 5x5 window
                double mean_val = 0.0;
                int count = 0;
                for (int di = -2; di <= 2; ++di) {
                    for (int dj = -2; dj <= 2; ++dj) {
                        if (ZI(i+di, j+dj) > 0) {
                            mean_val += ZI(i+di, j+dj);
                            count++;
                        }
                    }
                }
                if (count > 0) {
                    mean_val /= count;
                    double variance = 0.0;
                    for (int di = -2; di <= 2; ++di) {
                        for (int dj = -2; dj <= 2; ++dj) {
                            if (ZI(i+di, j+dj) > 0) {
                                variance += std::pow(ZI(i+di, j+dj) - mean_val, 2);
                            }
                        }
                    }
                    local_variance(i, j) = variance / count;
                }
            }
        }
    }

    // Adaptive edge threshold based on interpolation level
    double base_edge_threshold = 0.05 * arma::max(arma::max(grad_mag));
    double adaptive_factor = 1.0 + (layer_interpol - base_interpolation) * 0.1;
    double edge_threshold = base_edge_threshold * adaptive_factor;

    // Enhanced edge preservation with multi-scale filtering
    for (uint i = 2; i < ZI.n_rows - 2; ++i) {
        for (uint j = 2; j < ZI.n_cols - 2; ++j) {
            if (ZI(i, j) > 0) {
                // Combine gradient and variance information
                double edge_strength = grad_mag(i, j) + 0.3 * std::sqrt(local_variance(i, j));
                
                if (edge_strength > edge_threshold) {
                    // Strong edge - preserve original value with slight smoothing
                    double preservation_factor = std::min(1.0, edge_strength / edge_threshold);
                    
                    // Apply bilateral-like filtering
                    double weighted_sum = 0.0;
                    double weight_total = 0.0;
                    
                    for (int di = -2; di <= 2; ++di) {
                        for (int dj = -2; dj <= 2; ++dj) {
                            if (ZI(i+di, j+dj) > 0) {
                                double spatial_weight = std::exp(-(di*di + dj*dj) / (2.0 * 1.5 * 1.5));
                                double intensity_diff = std::abs(ZI(i+di, j+dj) - ZI(i, j));
                                double intensity_weight = std::exp(-intensity_diff * intensity_diff / (2.0 * 0.5 * 0.5));
                                double weight = spatial_weight * intensity_weight;
                                
                                weighted_sum += ZI(i+di, j+dj) * weight;
                                weight_total += weight;
                            }
                        }
                    }
                    
                    if (weight_total > 0) {
                        double filtered_value = weighted_sum / weight_total;
                        Zenhanced(i, j) = ZI(i, j) * preservation_factor + filtered_value * (1.0 - preservation_factor);
                    }
                } else {
                    // Smooth region - apply stronger filtering
                    double weighted_sum = 0.0;
                    double weight_total = 0.0;
                    
                    for (int di = -3; di <= 3; ++di) {
                        for (int dj = -3; dj <= 3; ++dj) {
                            if (i+di >= 0 && i+di < ZI.n_rows && j+dj >= 0 && j+dj < ZI.n_cols && ZI(i+di, j+dj) > 0) {
                                double distance = std::sqrt(di*di + dj*dj);
                                double weight = std::exp(-distance * distance / (2.0 * 2.0 * 2.0));
                                
                                weighted_sum += ZI(i+di, j+dj) * weight;
                                weight_total += weight;
                            }
                        }
                    }
                    
                    if (weight_total > 0) {
                        Zenhanced(i, j) = weighted_sum / weight_total;
                    }
                }
            }
        }
    }

    return Zenhanced;
}

// Variable interpolation function for VLP-16
arma::mat performVariableInterpolation(const arma::mat& Z, const arma::mat& Zz, int rows_img, int cols_img) {
    arma::vec X = arma::regspace(1, Z.n_cols);
    arma::vec Y = arma::regspace(1, Z.n_rows);
    
    // Calculate layer height
    int layer_height = rows_img / vlp16_layers;
    
    arma::mat ZI_combined = arma::zeros(rows_img * 20, cols_img); // Allocate enough space
    arma::mat ZzI_combined = arma::zeros(rows_img * 20, cols_img);
    
    int current_row = 0;
    
    for (int layer = 0; layer < vlp16_layers; ++layer) {
        int start_row = layer * layer_height;
        int end_row = std::min((layer + 1) * layer_height, (int)Z.n_rows);
        
        if (start_row >= end_row) continue;
        
        // Get interpolation value for this layer
        int layer_interpol = getInterpolationForLayer(layer);
        
        // Extract layer data
        arma::mat Z_layer = Z.rows(start_row, end_row - 1);
        arma::mat Zz_layer = Zz.rows(start_row, end_row - 1);
        
        // Create interpolation vectors for this layer
        arma::vec Y_layer = arma::regspace(start_row + 1, end_row);
        arma::vec YI_layer = arma::regspace(Y_layer.min(), 1.0/layer_interpol, Y_layer.max());
        
        // Interpolate this layer
        arma::mat ZI_layer, ZzI_layer;
        arma::interp2(X, Y_layer, Z_layer, X, YI_layer, ZI_layer, "linear");
        arma::interp2(X, Y_layer, Zz_layer, X, YI_layer, ZzI_layer, "linear");
        
        // Apply enhanced edge preservation for this layer
        ZI_layer = enhancedEdgePreservation(ZI_layer, ZzI_layer, layer_interpol);
        
        // Handle zeros in interpolation with layer-specific parameters
        arma::mat Zout_layer = ZI_layer;
        for (uint i = 0; i < ZI_layer.n_rows; i++) {
            for (uint j = 0; j < ZI_layer.n_cols; j++) {
                if (ZI_layer(i, j) == 0) {
                    int interpolation_range = layer_interpol / 2;
                    if (i + interpolation_range < ZI_layer.n_rows) {
                        for (int k = 1; k <= interpolation_range; k++) {
                            Zout_layer(i + k, j) = 0;
                        }
                    }
                    if (i > interpolation_range) {
                        for (int k = 1; k <= interpolation_range; k++) {
                            Zout_layer(i - k, j) = 0;
                        }
                    }
                }
            }
        }
        
        // Adaptive interpolation for missing data
        double density_threshold = 0.1 - (layer * 0.005); // Adjust threshold per layer
        int max_window_size = 7 + (layer_interpol - base_interpolation) / 2;
        int min_window_size = 3;
        
        for (uint i = 1; i < Zout_layer.n_rows - 1; ++i) {
            for (uint j = 1; j < Zout_layer.n_cols - 1; ++j) {
                if (Zout_layer(i, j) == 0) {
                    // Calculate local density
                    int valid_neighbors = 0;
                    for (int di = -2; di <= 2; ++di) {
                        for (int dj = -2; dj <= 2; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            if (ni >= 0 && nj >= 0 && ni < Zout_layer.n_rows && nj < Zout_layer.n_cols && Zout_layer(ni, nj) > 0) {
                                valid_neighbors++;
                            }
                        }
                    }
                    
                    // Determine window size
                    int window_size = (valid_neighbors < density_threshold * 25) ? max_window_size : min_window_size;
                    
                    // Multi-scale interpolation
                    double weighted_sum = 0.0;
                    double weight_total = 0.0;
                    
                    for (int di = -window_size; di <= window_size; ++di) {
                        for (int dj = -window_size; dj <= window_size; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            
                            if (ni >= 0 && nj >= 0 && ni < Zout_layer.n_rows && nj < Zout_layer.n_cols && Zout_layer(ni, nj) > 0) {
                                double distance = std::sqrt(di * di + dj * dj);
                                double weight = std::exp(-distance * distance / (2.0 * (window_size/2.0) * (window_size/2.0)));
                                weighted_sum += Zout_layer(ni, nj) * weight;
                                weight_total += weight;
                            }
                        }
                    }
                    
                    if (weight_total > 0) {
                        Zout_layer(i, j) = weighted_sum / weight_total;
                    }
                }
            }
        }
        
        // Store in combined matrix
        int layer_output_height = Zout_layer.n_rows;
        if (current_row + layer_output_height <= ZI_combined.n_rows) {
            ZI_combined.rows(current_row, current_row + layer_output_height - 1) = Zout_layer;
            ZzI_combined.rows(current_row, current_row + layer_output_height - 1) = ZzI_layer;
        }
        current_row += layer_output_height;
    }
    
    // Return properly sized matrix
    return ZI_combined.rows(0, current_row - 1);
}

///////////////////////////////////////callback
void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2, const ImageConstPtr& in_image)
{
    cv_bridge::CvImagePtr cv_ptr, color_pcl;
    try {
        cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
        color_pcl = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    //Conversion from sensor_msgs::PointCloud2 to pcl::PointCloud<T>
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*in_pc2, pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

    ////// filter point cloud 
    if (msg_pointCloud == NULL) return;

    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    for (int i = 0; i < (int)cloud_in->points.size(); i++) {
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);
        if (distance < minlen || distance > maxlen)
            continue;
        cloud_out->push_back(cloud_in->points[i]);
    }

    // Point cloud to image
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                                      pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                                      sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;

    arma::mat Z = arma::zeros(rows_img, cols_img);
    arma::mat Zz = arma::zeros(rows_img, cols_img);

    for (int i = 0; i < cols_img; ++i) {
        for (int j = 0; j < rows_img; ++j) {
            float r = rangeImage->getPoint(i, j).range;
            float zz = rangeImage->getPoint(i, j).z;

            if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz)) {
                continue;
            }
            Z.at(j, i) = r;
            Zz.at(j, i) = zz;
        }
    }

    // Perform variable interpolation
    arma::mat ZI = performVariableInterpolation(Z, Zz, rows_img, cols_img);

    // Continue with point cloud reconstruction
    PointCloud::Ptr point_cloud(new PointCloud);
    PointCloud::Ptr cloud(new PointCloud);

    // Variance filtering for point cloud
    if (f_pc) {
        arma::mat Zout = ZI;
        for (int layer = 0; layer < vlp16_layers; ++layer) {
            int layer_interpol = getInterpolationForLayer(layer);
            int layer_start = layer * (ZI.n_rows / vlp16_layers);
            int layer_end = std::min((layer + 1) * (ZI.n_rows / vlp16_layers), (int)ZI.n_rows);
            
            for (int i = layer_start; i < layer_end - layer_interpol; i += 1) {
                for (uint j = 0; j < ZI.n_cols - 5; j += 1) {
                    double promedio = 0;
                    double varianza = 0;
                    
                    for (int k = 0; k < layer_interpol; k += 1) {
                        if (i + k < ZI.n_rows) {
                            promedio += ZI(i + k, j);
                        }
                    }
                    promedio = promedio / layer_interpol;

                    for (int l = 0; l < layer_interpol; l++) {
                        if (i + l < ZI.n_rows) {
                            varianza += pow((ZI(i + l, j) - promedio), 2.0);
                        }
                    }

                    if (varianza > max_var) {
                        for (int m = 0; m < layer_interpol; m++) {
                            if (i + m < ZI.n_rows) {
                                Zout(i + m, j) = 0;
                            }
                        }
                    }
                }
            }
        }
        ZI = Zout;
    }

    // Range image to point cloud conversion
    int num_pc = 0;
    for (uint i = 0; i < ZI.n_rows - 5; i += 1) {
        for (uint j = 0; j < ZI.n_cols; j += 1) {
            float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));

            if (ang < min_FOV - M_PI/2.0 || ang > max_FOV - M_PI/2.0)
                continue;

            if (!(ZI(i, j) == 0)) {
                float pc_modulo = ZI(i, j);
                // Use stored Zz values from variable interpolation
                float zz_value = (i < Zz.n_rows && j < Zz.n_cols) ? Zz(i, j) : 0;
                float pc_x = sqrt(pow(pc_modulo, 2) - pow(zz_value, 2)) * cos(ang);
                float pc_y = sqrt(pow(pc_modulo, 2) - pow(zz_value, 2)) * sin(ang);

                float ang_x_lidar = 0.6 * M_PI / 180.0;

                Eigen::MatrixXf Lidar_matrix(3, 3);
                Eigen::MatrixXf result(3, 1);
                Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                                0, 1, 0,
                                -sin(ang_x_lidar), 0, cos(ang_x_lidar);

                result << pc_x, pc_y, zz_value;
                result = Lidar_matrix * result;

                pcl::PointXYZI point;
                point.x = result(0);
                point.y = result(1);
                point.z = result(2);
                point.intensity = pc_modulo;

                cloud->push_back(point);
                num_pc++;
            }
        }
    }

    // Rest of the processing (projection, coloring, etc.)
    PointCloud::Ptr P_out = cloud;

    Eigen::MatrixXf RTlc(4, 4);
    RTlc << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
            Rlc(1), Rlc(4), Rlc(7), Tlc(1),
            Rlc(2), Rlc(5), Rlc(8), Tlc(2),
            0, 0, 0, 1;

    int size_inter_Lidar = (int)P_out->points.size();
    unsigned int cols = in_image->width;
    unsigned int rows = in_image->height;

    uint px_data = 0;
    uint py_data = 0;

    pcl::PointXYZRGB point;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (int i = 0; i < size_inter_Lidar; i++) {
        Eigen::MatrixXf pc_matrix(4, 1);
        pc_matrix(0, 0) = -P_out->points[i].y;
        pc_matrix(1, 0) = -P_out->points[i].z;
        pc_matrix(2, 0) = P_out->points[i].x;
        pc_matrix(3, 0) = 1.0;

        Eigen::MatrixXf Lidar_cam = Mc * (RTlc * pc_matrix);

        px_data = (int)(Lidar_cam(0, 0) / Lidar_cam(2, 0));
        py_data = (int)(Lidar_cam(1, 0) / Lidar_cam(2, 0));

        if (px_data < 0.0 || px_data >= cols || py_data < 0.0 || py_data >= rows)
            continue;

        int color_dis_x = (int)(255 * ((P_out->points[i].x) / maxlen));
        int color_dis_z = (int)(255 * ((P_out->points[i].x) / 10.0));
        if (color_dis_z > 255)
            color_dis_z = 255;

        cv::Vec3b& color = color_pcl->image.at<cv::Vec3b>(py_data, px_data);

        point.x = P_out->points[i].x;
        point.y = P_out->points[i].y;
        point.z = P_out->points[i].z;
        point.r = (int)color[2];
        point.g = (int)color[1];
        point.b = (int)color[0];

        pc_color->points.push_back(point);

        cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1, CV_RGB(255 - color_dis_x, (int)(color_dis_z), color_dis_x), cv::FILLED);
    }

    pc_color->is_dense = true;
    pc_color->width = (int)pc_color->points.size();
    pc_color->height = 1;
    pc_color->header.frame_id = "velodyne";

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
    nh.getParam("/ang_Y_resolution", angular_resolution_y);
    
    // VLP-16 specific parameters
    nh.getParam("/vlp16_layers", vlp16_layers);
    nh.getParam("/base_interpolation", base_interpolation);
    nh.getParam("/interpolation_increment", interpolation_increment);

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

    message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic, 1);
    message_filters::Subscriber<Image> img_sub(nh, imgTopic, 1);

    typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);
    rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
    pc_pub = nh.advertise<PointCloud>("/points2", 1);

    ros::spin();
}