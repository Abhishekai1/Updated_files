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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <omp.h>  // For OpenMP parallelization

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/filters/statistical_outlier_removal.h>
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

float maxlen = 100.0;       
float minlen = 0.01;     
float max_FOV = 3.0;    
float min_FOV = 0.4;    

// Optimized parameters for denser output
float angular_resolution_x = 0.2f;  // Reduced for higher density
float angular_resolution_y = 1.5f;  // Reduced for higher density
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth = 100.0;
float min_depth = 8.0;
double max_var = 75.0;  // Increased tolerance for far regions

float interpol_value = 15.0;  // Optimized interpolation value

bool f_pc = true; 

// input topics 
std::string imgTopic = "/camera/color/image_raw";
std::string pcTopic = "/velodyne_points";

//matrix calibration lidar and camera
Eigen::MatrixXf Tlc(3,1); 
Eigen::MatrixXf Rlc(3,3); 
Eigen::MatrixXf Mc(3,4);  

// range image parametros
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

// Optimized single-pass interpolation with adaptive density
arma::mat optimizedInterpolation(const arma::mat& Z, const arma::mat& Zz, float max_distance) {
    arma::mat result = Z;
    
    // Pre-compute distance map for efficiency
    arma::mat distance_map = arma::zeros(Z.n_rows, Z.n_cols);
    
    #pragma omp parallel for collapse(2)
    for (uint i = 0; i < Z.n_rows; ++i) {
        for (uint j = 0; j < Z.n_cols; ++j) {
            if (Z(i, j) > 0) {
                distance_map(i, j) = Z(i, j);
            }
        }
    }
    
    // Single-pass adaptive interpolation
    #pragma omp parallel for collapse(2)
    for (uint i = 2; i < Z.n_rows - 2; ++i) {
        for (uint j = 2; j < Z.n_cols - 2; ++j) {
            if (Z(i, j) == 0) {  // Missing data point
                
                // Quick neighborhood analysis
                double local_avg_distance = 0.0;
                int valid_count = 0;
                double weighted_sum = 0.0;
                double weight_total = 0.0;
                
                // Determine adaptive window size based on local density
                int base_window = 2;
                int adaptive_window = base_window;
                
                // Quick scan for local characteristics
                for (int di = -base_window; di <= base_window; ++di) {
                    for (int dj = -base_window; dj <= base_window; ++dj) {
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni >= 0 && nj >= 0 && ni < Z.n_rows && nj < Z.n_cols && Z(ni, nj) > 0) {
                            local_avg_distance += Z(ni, nj);
                            valid_count++;
                        }
                    }
                }
                
                if (valid_count > 0) {
                    local_avg_distance /= valid_count;
                    
                    // Adaptive window sizing
                    if (local_avg_distance > max_distance * 0.6) {
                        adaptive_window = 6;  // Larger window for far regions
                    } else if (local_avg_distance > max_distance * 0.3) {
                        adaptive_window = 4;  // Medium window
                    } else {
                        adaptive_window = 3;  // Small window for near regions
                    }
                    
                    // Efficient weighted interpolation
                    for (int di = -adaptive_window; di <= adaptive_window; ++di) {
                        for (int dj = -adaptive_window; dj <= adaptive_window; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            
                            if (ni >= 0 && nj >= 0 && ni < Z.n_rows && nj < Z.n_cols && Z(ni, nj) > 0) {
                                double distance = std::sqrt(di * di + dj * dj);
                                double weight;
                                
                                if (local_avg_distance > max_distance * 0.5) {
                                    // Gaussian weighting for far regions
                                    weight = std::exp(-distance * distance / (adaptive_window * adaptive_window));
                                } else {
                                    // Inverse distance weighting for near regions
                                    weight = 1.0 / (distance + 0.1);
                                }
                                
                                weighted_sum += Z(ni, nj) * weight;
                                weight_total += weight;
                            }
                        }
                    }
                    
                    if (weight_total > 0) {
                        result(i, j) = weighted_sum / weight_total;
                    }
                }
            }
        }
    }
    
    return result;
}

// Optimized dense point generation
PointCloud::Ptr generateDensePointCloud(const arma::mat& ZI, const arma::mat& ZzI, 
                                        float maxlen, float min_FOV, float max_FOV) {
    PointCloud::Ptr dense_cloud(new PointCloud);
    
    // Pre-allocate for efficiency
    dense_cloud->points.reserve(ZI.n_rows * ZI.n_cols);
    
    float ang_x_lidar = 0.6 * M_PI / 180.0;
    
    Eigen::Matrix3f Lidar_matrix;
    Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                    0, 1, 0,
                    -sin(ang_x_lidar), 0, cos(ang_x_lidar);
    
    #pragma omp parallel for collapse(2)
    for (uint i = 0; i < ZI.n_rows; i++) {
        for (uint j = 0; j < ZI.n_cols; j++) {
            if (ZI(i, j) > minlen && ZI(i, j) < maxlen) {
                
                float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));
                
                if (ang >= min_FOV - M_PI/2.0 && ang <= max_FOV - M_PI/2.0) {
                    
                    float pc_modulo = ZI(i, j);
                    float pc_x = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * cos(ang);
                    float pc_y = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * sin(ang);
                    
                    Eigen::Vector3f point_vec(pc_x, pc_y, ZzI(i, j));
                    point_vec = Lidar_matrix * point_vec;
                    
                    pcl::PointXYZI point;
                    point.x = point_vec(0);
                    point.y = point_vec(1);
                    point.z = point_vec(2);
                    point.intensity = pc_modulo / maxlen * 255.0;  // Normalized intensity
                    
                    #pragma omp critical
                    {
                        dense_cloud->points.push_back(point);
                        
                        // Add sub-pixel interpolated points for ultra-dense visualization
                        if (pc_modulo > maxlen * 0.3) {  // Only for medium to far ranges
                            // Add 4 sub-points around each main point
                            for (int sub_i = -1; sub_i <= 1; sub_i += 2) {
                                for (int sub_j = -1; sub_j <= 1; sub_j += 2) {
                                    pcl::PointXYZI sub_point;
                                    sub_point.x = point_vec(0) + sub_i * 0.05;
                                    sub_point.y = point_vec(1) + sub_j * 0.05;
                                    sub_point.z = point_vec(2);
                                    sub_point.intensity = point.intensity * 0.8;
                                    dense_cloud->points.push_back(sub_point);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    dense_cloud->width = dense_cloud->points.size();
    dense_cloud->height = 1;
    dense_cloud->is_dense = false;
    
    return dense_cloud;
}

// Optimized noise reduction with parallel processing
arma::mat optimizedNoiseReduction(const arma::mat& ZI, double max_var, float maxlen, float interpol_value) {
    arma::mat result = ZI;
    
    #pragma omp parallel for
    for (uint i = 0; i < ((ZI.n_rows-1) / interpol_value); i++) {
        for (uint j = 0; j < ZI.n_cols - 5; j++) {
            double sum = 0;
            double sum_sq = 0;
            double avg_distance = 0;
            int valid_points = 0;
            
            // Calculate statistics in one pass
            for (uint k = 0; k < interpol_value; k++) {
                uint row_idx = (i * interpol_value) + k;
                if (row_idx < ZI.n_rows && ZI(row_idx, j) > 0) {
                    double val = ZI(row_idx, j);
                    sum += val;
                    sum_sq += val * val;
                    avg_distance += val;
                    valid_points++;
                }
            }
            
            if (valid_points > 2) {  // Need at least 3 points for meaningful variance
                double mean = sum / valid_points;
                double variance = (sum_sq - sum * mean) / valid_points;
                avg_distance /= valid_points;
                
                // Adaptive variance threshold
                double adaptive_threshold = max_var;
                if (avg_distance > maxlen * 0.6) {
                    adaptive_threshold = max_var * 1.5;  // More lenient for far regions
                }
                
                if (variance > adaptive_threshold) {
                    for (uint m = 0; m < interpol_value; m++) {
                        uint row_idx = (i * interpol_value) + m;
                        if (row_idx < result.n_rows) {
                            result(row_idx, j) = 0;
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2, 
              const ImageConstPtr& in_image) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cv_bridge::CvImagePtr cv_ptr, color_pcl;
    try {
        cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
        color_pcl = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Convert point cloud
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*in_pc2, pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

    if (msg_pointCloud == NULL) return;

    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);

    // Remove NaN points
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);
    
    // Filter by distance - parallelized
    cloud_out->points.reserve(cloud_in->points.size());
    
    #pragma omp parallel for
    for (int i = 0; i < (int)cloud_in->points.size(); i++) {
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + 
                              cloud_in->points[i].y * cloud_in->points[i].y);     
        if (distance >= minlen && distance <= maxlen) {
            #pragma omp critical
            {
                cloud_out->push_back(cloud_in->points[i]);
            }
        }
    }

    // Create range image
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, 
                                                      pcl::deg2rad(angular_resolution_x), 
                                                      pcl::deg2rad(angular_resolution_y),
                                                      pcl::deg2rad(max_angle_width), 
                                                      pcl::deg2rad(max_angle_height),
                                                      sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;

    arma::mat Z = arma::zeros(rows_img, cols_img);         
    arma::mat Zz = arma::zeros(rows_img, cols_img);       

    // Fill range image matrices - parallelized
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < cols_img; ++i) {
        for (int j = 0; j < rows_img; ++j) {
            float r = rangeImage->getPoint(i, j).range;     
            float zz = rangeImage->getPoint(i, j).z; 
           
            if (!std::isinf(r) && !std::isnan(r) && !std::isnan(zz) && 
                r >= minlen && r <= maxlen) {
                Z(j, i) = r;   
                Zz(j, i) = zz;
            }
        }
    }

    // Enhanced interpolation
    arma::vec X = arma::regspace(1, Z.n_cols);
    arma::vec Y = arma::regspace(1, Z.n_rows);
    arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
    arma::vec YI = arma::regspace(Y.min(), 1.0/interpol_value, Y.max());

    arma::mat ZI, ZzI;
    arma::interp2(X, Y, Z, XI, YI, ZI, "linear");  
    arma::interp2(X, Y, Zz, XI, YI, ZzI, "linear");  

    // Apply optimized interpolation
    ZI = optimizedInterpolation(ZI, ZzI, maxlen);

    // Apply noise reduction if enabled
    if (f_pc) {
        ZI = optimizedNoiseReduction(ZI, max_var, maxlen, interpol_value);
    }

    // Generate dense point cloud
    PointCloud::Ptr dense_cloud = generateDensePointCloud(ZI, ZzI, maxlen, min_FOV, max_FOV);

    // Transform to camera coordinates and colorize
    Eigen::Matrix4f RTlc;
    RTlc.block<3,3>(0,0) = Rlc;
    RTlc.block<3,1>(0,3) = Tlc;
    RTlc.row(3) << 0, 0, 0, 1;

    unsigned int cols = in_image->width;
    unsigned int rows = in_image->height;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color(new pcl::PointCloud<pcl::PointXYZRGB>);
    pc_color->points.reserve(dense_cloud->points.size());

    #pragma omp parallel for
    for (int i = 0; i < (int)dense_cloud->points.size(); i++) {
        Eigen::Vector4f pc_matrix(-dense_cloud->points[i].y, 
                                  -dense_cloud->points[i].z, 
                                   dense_cloud->points[i].x, 1.0);

        Eigen::Vector3f Lidar_cam = Mc * (RTlc * pc_matrix);

        int px_data = (int)(Lidar_cam(0) / Lidar_cam(2));
        int py_data = (int)(Lidar_cam(1) / Lidar_cam(2));
        
        if (px_data >= 0 && px_data < cols && py_data >= 0 && py_data < rows) {
            
            pcl::PointXYZRGB point;
            point.x = dense_cloud->points[i].x;
            point.y = dense_cloud->points[i].y;
            point.z = dense_cloud->points[i].z;
            
            cv::Vec3b & color = color_pcl->image.at<cv::Vec3b>(py_data, px_data);
            point.r = (int)color[2]; 
            point.g = (int)color[1]; 
            point.b = (int)color[0];
            
            #pragma omp critical
            {
                pc_color->points.push_back(point);
            }
            
            // Enhanced visualization with distance-adaptive point sizes
            double distance = sqrt(dense_cloud->points[i].x * dense_cloud->points[i].x + 
                                  dense_cloud->points[i].y * dense_cloud->points[i].y + 
                                  dense_cloud->points[i].z * dense_cloud->points[i].z);
            
            int color_dis_x = (int)(255 * (distance / maxlen));
            int circle_radius = (distance > maxlen * 0.6) ? 2 : 1;
            
            cv::circle(cv_ptr->image, cv::Point(px_data, py_data), circle_radius, 
                      CV_RGB(255 - color_dis_x, color_dis_x, 128), cv::FILLED);
        }
    }

    pc_color->width = pc_color->points.size();
    pc_color->height = 1;
    pc_color->is_dense = true;
    pc_color->header.frame_id = "velodyne";

    // Publish results
    pcOnimg_pub.publish(cv_ptr->toImageMsg());
    pc_pub.publish(pc_color);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    ROS_INFO("Processing time: %ld ms, Points generated: %zu", duration.count(), pc_color->points.size());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "optimized_pointCloudOnImage");
    ros::NodeHandle nh;  

    // Set OpenMP threads for optimal performance
    omp_set_num_threads(std::min(8, (int)std::thread::hardware_concurrency()));

    // Load Parameters
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

    message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic, 1);
    message_filters::Subscriber<Image> img_sub(nh, imgTopic, 1);

    typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));
    
    pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);
    rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
    pc_pub = nh.advertise<PointCloud>("/points2", 1);

    ROS_INFO("Optimized LiDAR-Camera Fusion Node Started");
    ros::spin();
    
    return 0;
}
