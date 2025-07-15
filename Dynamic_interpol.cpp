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

float maxlen = 100.0;       
float minlen = 0.01;     
float max_FOV = 3.0;    
float min_FOV = 0.4;    

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

// Structure to hold local interpolation parameters
struct InterpolationParams {
    float factor;
    int window_size;
    float edge_threshold;
    float variance_threshold;
};

// Adaptive interpolation function based on local point density and distance
InterpolationParams getAdaptiveInterpolation(float distance, float density, float variance) {
    InterpolationParams params;
    
    // Base interpolation factor based on distance
    float base_factor;
    if (distance < 5.0f) {
        base_factor = 3.0f;
    } else if (distance < 10.0f) {
        base_factor = 6.0f;
    } else if (distance < 20.0f) {
        base_factor = 9.0f;
    } else if (distance < 40.0f) {
        base_factor = 12.0f;
    } else {
        base_factor = 15.0f;
    }
    
    // Adjust based on local density (higher density = less interpolation needed)
    float density_factor = std::max(0.5f, std::min(2.0f, 1.0f / (density + 0.1f)));
    
    // Adjust based on variance (higher variance = more conservative interpolation)
    float variance_factor = std::max(0.7f, std::min(1.5f, 1.0f / (variance / 10.0f + 0.1f)));
    
    params.factor = base_factor * density_factor * variance_factor;
    params.window_size = static_cast<int>(std::max(3.0f, std::min(15.0f, params.factor * 0.8f)));
    params.edge_threshold = 0.1f / (distance * 0.1f + 1.0f);
    params.variance_threshold = max_var * (1.0f + distance * 0.02f);
    
    return params;
}

// Calculate local point density in a region
float calculateLocalDensity(const arma::mat& Z, int row, int col, int window_size) {
    int valid_points = 0;
    int total_points = 0;
    
    int start_row = std::max(0, row - window_size);
    int end_row = std::min(static_cast<int>(Z.n_rows) - 1, row + window_size);
    int start_col = std::max(0, col - window_size);
    int end_col = std::min(static_cast<int>(Z.n_cols) - 1, col + window_size);
    
    for (int i = start_row; i <= end_row; ++i) {
        for (int j = start_col; j <= end_col; ++j) {
            total_points++;
            if (Z(i, j) > minlen && Z(i, j) < maxlen) {
                valid_points++;
            }
        }
    }
    
    return total_points > 0 ? static_cast<float>(valid_points) / total_points : 0.0f;
}

// Calculate local variance
float calculateLocalVariance(const arma::mat& Z, int row, int col, int window_size) {
    std::vector<float> values;
    float sum = 0.0f;
    
    int start_row = std::max(0, row - window_size);
    int end_row = std::min(static_cast<int>(Z.n_rows) - 1, row + window_size);
    int start_col = std::max(0, col - window_size);
    int end_col = std::min(static_cast<int>(Z.n_cols) - 1, col + window_size);
    
    for (int i = start_row; i <= end_row; ++i) {
        for (int j = start_col; j <= end_col; ++j) {
            if (Z(i, j) > minlen && Z(i, j) < maxlen) {
                values.push_back(Z(i, j));
                sum += Z(i, j);
            }
        }
    }
    
    if (values.size() < 2) return 0.0f;
    
    float mean = sum / values.size();
    float variance = 0.0f;
    
    for (float val : values) {
        variance += (val - mean) * (val - mean);
    }
    
    return variance / (values.size() - 1);
}

// Adaptive interpolation function
void adaptiveInterpolation(arma::mat& ZI, const arma::mat& ZzI, const arma::mat& original_Z) {
    // Create density and variance maps
    arma::mat density_map(ZI.n_rows, ZI.n_cols, arma::fill::zeros);
    arma::mat variance_map(ZI.n_rows, ZI.n_cols, arma::fill::zeros);
    
    // Calculate local properties
    for (uint i = 0; i < ZI.n_rows; ++i) {
        for (uint j = 0; j < ZI.n_cols; ++j) {
            density_map(i, j) = calculateLocalDensity(original_Z, i, j, 5);
            variance_map(i, j) = calculateLocalVariance(original_Z, i, j, 3);
        }
    }
    
    // Adaptive interpolation
    for (uint i = 1; i < ZI.n_rows - 1; ++i) {
        for (uint j = 1; j < ZI.n_cols - 1; ++j) {
            if (ZI(i, j) == 0) {  // Missing data point
                float local_density = density_map(i, j);
                float local_variance = variance_map(i, j);
                float avg_distance = 0.0f;
                int count = 0;
                
                // Calculate average distance in local neighborhood
                for (int di = -2; di <= 2; ++di) {
                    for (int dj = -2; dj <= 2; ++dj) {
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                            avg_distance += ZI(ni, nj);
                            count++;
                        }
                    }
                }
                
                if (count > 0) {
                    avg_distance /= count;
                    
                    // Get adaptive parameters
                    InterpolationParams params = getAdaptiveInterpolation(avg_distance, local_density, local_variance);
                    
                    // Perform weighted interpolation
                    double weighted_sum = 0.0;
                    double weight_total = 0.0;
                    
                    for (int di = -params.window_size; di <= params.window_size; ++di) {
                        for (int dj = -params.window_size; dj <= params.window_size; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            
                            if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                                double distance = std::sqrt(di * di + dj * dj);
                                double weight = 1.0 / (distance * params.factor + 1e-6);
                                
                                // Additional weight based on local density
                                weight *= (1.0 + local_density);
                                
                                // Reduce weight for high variance areas
                                if (local_variance > params.variance_threshold) {
                                    weight *= 0.5;
                                }
                                
                                weighted_sum += ZI(ni, nj) * weight;
                                weight_total += weight;
                            }
                        }
                    }
                    
                    if (weight_total > 0) {
                        ZI(i, j) = weighted_sum / weight_total;
                    }
                }
            }
        }
    }
}

// Dynamic edge-aware filtering
void dynamicEdgePreservation(arma::mat& ZI, const arma::mat& density_map) {
    arma::mat grad_x = arma::zeros(ZI.n_rows, ZI.n_cols);
    arma::mat grad_y = arma::zeros(ZI.n_rows, ZI.n_cols);
    arma::mat grad_mag = arma::zeros(ZI.n_rows, ZI.n_cols);
    
    // Calculate gradients
    for (uint i = 1; i < ZI.n_rows - 1; ++i) {
        for (uint j = 1; j < ZI.n_cols - 1; ++j) {
            if (ZI(i, j) > 0) {
                grad_x(i, j) = (ZI(i, j + 1) - ZI(i, j - 1)) * 0.5;
                grad_y(i, j) = (ZI(i + 1, j) - ZI(i - 1, j)) * 0.5;
                grad_mag(i, j) = std::sqrt(grad_x(i, j) * grad_x(i, j) + grad_y(i, j) * grad_y(i, j));
            }
        }
    }
    
    // Apply dynamic edge preservation
    for (uint i = 1; i < ZI.n_rows - 1; ++i) {
        for (uint j = 1; j < ZI.n_cols - 1; ++j) {
            float local_density = density_map(i, j);
            float adaptive_threshold = 0.1f * arma::max(arma::max(grad_mag)) * (1.0f + local_density);
            
            if (grad_mag(i, j) > adaptive_threshold) {
                // Preserve edges more in dense areas
                float preservation_factor = std::min(0.9f, 0.5f + local_density * 0.4f);
                
                // Apply bilateral-like filtering
                double weighted_sum = 0.0;
                double weight_total = 0.0;
                
                for (int di = -2; di <= 2; ++di) {
                    for (int dj = -2; dj <= 2; ++dj) {
                        int ni = i + di;
                        int nj = j + dj;
                        
                        if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                            double spatial_dist = std::sqrt(di * di + dj * dj);
                            double intensity_diff = std::abs(ZI(ni, nj) - ZI(i, j));
                            
                            double spatial_weight = std::exp(-spatial_dist * spatial_dist / (2.0 * 1.0 * 1.0));
                            double intensity_weight = std::exp(-intensity_diff * intensity_diff / (2.0 * 2.0 * 2.0));
                            
                            double weight = spatial_weight * intensity_weight;
                            weighted_sum += ZI(ni, nj) * weight;
                            weight_total += weight;
                        }
                    }
                }
                
                if (weight_total > 0) {
                    float filtered_value = weighted_sum / weight_total;
                    ZI(i, j) = ZI(i, j) * preservation_factor + filtered_value * (1.0f - preservation_factor);
                }
            }
        }
    }
}

///////////////////////////////////////callback
void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2, const ImageConstPtr& in_image)
{
    cv_bridge::CvImagePtr cv_ptr, color_pcl;
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
    pcl_conversions::toPCL(*in_pc2, pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

    ////// filter point cloud 
    if (msg_pointCloud == NULL) return;

    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    for (int i = 0; i < (int)cloud_in->points.size(); i++)
    {
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);
        if (distance < minlen || distance > maxlen)
            continue;
        cloud_out->push_back(cloud_in->points[i]);
    }

    //                                                  point cloud to image 
    //============================================================================================================
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                                      pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                                      sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;

    arma::mat Z;  
    arma::mat Zz; 

    Z.zeros(rows_img, cols_img);
    Zz.zeros(rows_img, cols_img);

    for (int i = 0; i < cols_img; ++i)
        for (int j = 0; j < rows_img; ++j)
        {
            float r = rangeImage->getPoint(i, j).range;
            float zz = rangeImage->getPoint(i, j).z;

            if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz)) {
                continue;
            }
            Z.at(j, i) = r;
            Zz.at(j, i) = zz;
        }

    ////////////////////////////////////////////// Dynamic interpolation
    //============================================================================================================
    
    // Store original Z for reference
    arma::mat original_Z = Z;
    
    // Create density map
    arma::mat density_map(Z.n_rows, Z.n_cols, arma::fill::zeros);
    for (uint i = 0; i < Z.n_rows; ++i) {
        for (uint j = 0; j < Z.n_cols; ++j) {
            density_map(i, j) = calculateLocalDensity(Z, i, j, 5);
        }
    }
    
    // Adaptive interpolation based on local characteristics
    arma::vec X = arma::regspace(1, Z.n_cols);
    arma::vec Y = arma::regspace(1, Z.n_rows);

    // Dynamic interpolation factor based on overall point cloud density
    float overall_density = arma::mean(arma::mean(density_map));
    float adaptive_y_factor = std::max(5.0f, std::min(25.0f, 15.0f / (overall_density + 0.1f)));

    arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
    arma::vec YI = arma::regspace(Y.min(), 1.0 / adaptive_y_factor, Y.max());

    arma::mat ZI, ZzI;
    arma::interp2(X, Y, Z, XI, YI, ZI, "linear");
    arma::interp2(X, Y, Zz, XI, YI, ZzI, "linear");

    // Apply adaptive interpolation to fill missing data
    adaptiveInterpolation(ZI, ZzI, original_Z);

    // Apply dynamic edge preservation
    dynamicEdgePreservation(ZI, density_map);

    // Handle interpolation artifacts with dynamic masking
    arma::mat Zout = ZI;
    for (uint i = 0; i < ZI.n_rows; i++) {
        for (uint j = 0; j < ZI.n_cols; j++) {
            if (ZI(i, j) == 0) {
                float local_density = density_map(std::min(i, density_map.n_rows - 1), std::min(j, density_map.n_cols - 1));
                int mask_size = static_cast<int>(adaptive_y_factor * (1.0f - local_density * 0.5f));
                
                if (i + mask_size < ZI.n_rows) {
                    for (int k = 1; k <= mask_size; k++) {
                        Zout(i + k, j) = 0;
                    }
                }
                if (i > mask_size) {
                    for (int k = 1; k <= mask_size; k++) {
                        Zout(i - k, j) = 0;
                    }
                }
            }
        }
    }
    ZI = Zout;

    // Rest of the processing remains the same...
    // [Continue with the point cloud reconstruction and publishing code]
    
    if (f_pc) {
        // Dynamic variance filtering
        for (uint i = 0; i < ((ZI.n_rows - 1) / static_cast<uint>(adaptive_y_factor)); i += 1)
            for (uint j = 0; j < ZI.n_cols - 5; j += 1)
            {
                double promedio = 0;
                double varianza = 0;
                int interpol_value = static_cast<int>(adaptive_y_factor);
                
                for (uint k = 0; k < interpol_value; k += 1)
                    promedio = promedio + ZI((i * interpol_value) + k, j);

                promedio = promedio / interpol_value;

                for (uint l = 0; l < interpol_value; l++)
                    varianza = varianza + pow((ZI((i * interpol_value) + l, j) - promedio), 2.0);

                // Dynamic variance threshold
                float local_density = density_map(std::min(i * interpol_value, density_map.n_rows - 1), std::min(j, density_map.n_cols - 1));
                double dynamic_max_var = max_var * (1.0 + local_density * 0.5);

                if (varianza > dynamic_max_var)
                    for (uint m = 0; m < interpol_value; m++)
                        Zout((i * interpol_value) + m, j) = 0;
            }
        ZI = Zout;
    }

    // Continue with point cloud reconstruction...
    PointCloud::Ptr point_cloud(new PointCloud);
    PointCloud::Ptr cloud(new PointCloud);
    
    int num_pc = 0;
    int interpol_value = static_cast<int>(adaptive_y_factor);
    
    for (uint i = 0; i < ZI.n_rows - interpol_value; i += 1)
    {
        for (uint j = 0; j < ZI.n_cols; j += 1)
        {
            float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));

            if (ang < min_FOV - M_PI / 2.0 || ang > max_FOV - M_PI / 2.0)
                continue;

            if (!(Zout(i, j) == 0))
            {
                float pc_modulo = Zout(i, j);
                float pc_x = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * cos(ang);
                float pc_y = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * sin(ang);

                float ang_x_lidar = 0.6 * M_PI / 180.0;

                Eigen::MatrixXf Lidar_matrix(3, 3);
                Eigen::MatrixXf result(3, 1);
                Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                    0, 1, 0,
                    -sin(ang_x_lidar), 0, cos(ang_x_lidar);

                result << pc_x,
                    pc_y,
                    ZzI(i, j);

                result = Lidar_matrix * result;

                pcl::PointXYZI point;
                point.x = result(0);
                point.y = result(1);
                point.z = result(2);
                point.intensity = 1.0;

                cloud->push_back(point);
                num_pc++;
            }
        }
    }

    // Continue with camera projection and publishing...
    PointCloud::Ptr P_out = cloud;

    Eigen::MatrixXf RTlc(4, 4);
    RTlc << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
        Rlc(1), Rlc(4), Rlc(7), Tlc(1),
        Rlc(2), Rlc(5), Rlc(8), Tlc(2),
        0, 0, 0, 1;

    int size_inter_Lidar = (int)P_out->points.size();

    Eigen::MatrixXf Lidar_cam(3, 1);
    Eigen::MatrixXf pc_matrix(4, 1);

    unsigned int cols = in_image->width;
    unsigned int rows = in_image->height;

    uint px_data = 0; uint py_data = 0;

    pcl::PointXYZRGB point;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (int i = 0; i < size_inter_Lidar; i++)
    {
        pc_matrix(0, 0) = -P_out->points[i].y;
        pc_matrix(1, 0) = -P_out->points[i].z;
        pc_matrix(2, 0) = P_out->points[i].x;
        pc_matrix(3, 0) = 1.0;

        Lidar_cam = Mc * (RTlc * pc_matrix);

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

    XmlRpc::XmlRpcValue param;

    nh.getParam("/matrix_file/tlc", param);
    Tlc << (double)param[0],
        (double)param[1],
        (double)param[2];

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

  pc_pub = nh.advertise<PointCloud> ("/points2", 1);  

  ros::spin();
}
