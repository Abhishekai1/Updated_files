#include <ros/ros.h>
#include <limits>
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
#include <armadillo>
#include <Eigen/Dense>
#include <math.h>
#include <chrono>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

typedef std::chrono::high_resolution_clock Clock;

using namespace Eigen;
using namespace sensor_msgs;
using namespace std;

// Updated to use PointXYZIR for ring information
typedef pcl::PointCloud<pcl::PointXYZIR> PointCloud;

// Publisher
ros::Publisher pc_pub;
ros::Publisher imgD_pub;

float maxlen = 100.0;     // Maximum LiDAR distance
float minlen = 0.01;      // Minimum LiDAR distance
float max_FOV = 3.0;      // Camera max FOV in radians
float min_FOV = 0.4;      // Camera min FOV in radians

// Parameters for point cloud to image conversion
float angular_resolution_x = 1.0f;
float angular_resolution_y = 1.5f;
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;

float interpol_value = 15.0;
float ang_x_lidar = 0.6 * M_PI / 180.0;
double max_var = 50.0;
bool f_pc = true;
int num_layers = 16; // Number of LiDAR layers (e.g., 16 for VLP-16)
int min_window_size = 3; // Minimum interpolation kernel size
int max_window_size = 9; // Maximum interpolation kernel size

// Input topic
std::string pcTopic = "/velodyne_points";

// Range image parameters
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

// Function to determine kernel size based on layer
int getKernelSize(int layer) {
    return min_window_size + (layer * (max_window_size - min_window_size) / (num_layers - 1));
}

void callback(const PointCloud::ConstPtr& msg_pointCloud)
{
    if (msg_pointCloud == NULL) return;

    // Filter point cloud
    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    float max_z = 0, min_z = std::numeric_limits<float>::infinity();
    float max_dis = 0, min_dis = std::numeric_limits<float>::infinity();

    // Split points into layers
    std::vector<std::vector<pcl::PointXYZIR>> layers(num_layers);
    for (const auto& pt : cloud_in->points) {
        double distance = sqrt(pt.x * pt.x + pt.y * pt.y);
        if (distance < minlen || distance > maxlen) continue;

        int ring = pt.ring;
        if (ring >= 0 && ring < num_layers) {
            layers[ring].push_back(pt);
            cloud_out->push_back(pt);
            if (pt.z > max_z) max_z = pt.z;
            if (pt.z < min_z) min_z = pt.z;
            if (distance > max_dis) max_dis = distance;
            if (distance < min_dis) min_dis = distance;
        }
    }

    // Create range images per layer
    std::vector<boost::shared_ptr<pcl::RangeImageSpherical>> rangeImages(num_layers);
    std::vector<arma::mat> Z_layers(num_layers);
    std::vector<arma::mat> Zz_layers(num_layers);
    float max_depth = 0.0;
    float min_depth = std::numeric_limits<float>::infinity();

    for (int ring = 0; ring < num_layers; ++ring) {
        if (layers[ring].empty()) continue;

        rangeImages[ring] = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
        Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
        PointCloud::Ptr layer_cloud(new PointCloud);
        *layer_cloud = PointCloud(layers[ring].begin(), layers[ring].end());

        rangeImages[ring]->createFromPointCloud(*layer_cloud, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                                pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                                sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

        int cols_img = rangeImages[ring]->width;
        int rows_img = 1; // Single row per layer
        Z_layers[ring].zeros(rows_img, cols_img);
        Zz_layers[ring].zeros(rows_img, cols_img);

        for (int i = 0; i < cols_img; ++i) {
            float r = rangeImages[ring]->getPoint(i, 0).range;
            float zz = rangeImages[ring]->getPoint(i, 0).z;
            if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz)) continue;
            Z_layers[ring].at(0, i) = r;
            Zz_layers[ring].at(0, i) = zz;
            if (r > max_depth) max_depth = r;
            if (r < min_depth) min_depth = r;
        }
    }

    // Dynamic interpolation per layer
    std::vector<arma::mat> ZI_layers(num_layers);
    std::vector<arma::mat> ZzI_layers(num_layers);

    for (int ring = 0; ring < num_layers; ++ring) {
        if (Z_layers[ring].n_elem == 0) continue;

        arma::vec X = arma::regspace(1, Z_layers[ring].n_cols);
        arma::vec Y = arma::regspace(1, 1);
        arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
        arma::vec YI = arma::regspace(Y.min(), 1.0 / interpol_value, Y.max());

        arma::mat ZI, ZzI;
        arma::interp2(X, Y, Z_layers[ring], XI, YI, ZI, "lineal");
        arma::interp2(X, Y, Zz_layers[ring], XI, YI, ZzI, "lineal");

        arma::mat Zout = ZI;
        int window_size = getKernelSize(ring);
        double density_threshold = 0.05;

        // Handle zeros and interpolate
        for (uint i = 0; i < ZI.n_rows; ++i) {
            for (uint j = 0; j < ZI.n_cols; ++j) {
                if (ZI(i, j) == 0) {
                    if (i + interpol_value < ZI.n_rows) {
                        for (int k = 1; k <= interpol_value; k++) {
                            Zout(i + k, j) = 0;
                        }
                    }
                    if (i > interpol_value) {
                        for (int k = 1; k <= interpol_value; k++) {
                            Zout(i - k, j) = 0;
                        }
                    }

                    double weighted_sum = 0.0;
                    double weight_total = 0.0;
                    int valid_neighbors = 0;

                    for (int di = -1; di <= 1; ++di) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                                valid_neighbors++;
                            }
                        }
                    }

                    for (int di = -window_size; di <= window_size; ++di) {
                        for (int dj = -window_size; dj <= window_size; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                                double distance = std::sqrt(di * di + dj * dj);
                                double weight = 1.0 / (distance + 1e-6);
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

        // Edge preservation
        arma::mat Zenhanced = ZI;
        arma::mat grad_x = arma::zeros(ZI.n_rows, ZI.n_cols);
        arma::mat grad_y = arma::zeros(ZI.n_rows, ZI.n_cols);
        arma::mat grad_mag = arma::zeros(ZI.n_rows, ZI.n_cols);

        for (uint i = 1; i < ZI.n_rows - 1; ++i) {
            for (uint j = 1; j < ZI.n_cols - 1; ++j) {
                if (ZI(i, j) > 0) {
                    grad_x(i, j) = (ZI(i, j + 1) - ZI(i, j - 1)) * 0.5;
                    grad_y(i, j) = (ZI(i + 1, j) - ZI(i - 1, j)) * 0.5;
                    grad_mag(i, j) = std::sqrt(grad_x(i, j) * grad_x(i, j) + grad_y(i, j) * grad_y(i, j));
                }
            }
        }

        double edge_threshold = 0.1 * arma::max(arma::max(grad_mag));
        for (uint i = 1; i < ZI.n_rows - 1; ++i) {
            for (uint j = 1; j < ZI.n_cols - 1; ++j) {
                if (grad_mag(i, j) > edge_threshold) {
                    double weight = std::max(0.0, 1.0 - grad_mag(i, j) / edge_threshold);
                    Zenhanced(i, j) = ZI(i, j) * weight + Zenhanced(i, j) * (1 - weight);
                }
            }
        }

        ZI_layers[ring] = Zenhanced;
        ZzI_layers[ring] = ZzI;
    }

    // Variance filtering
    if (f_pc) {
        for (int ring = 0; ring < num_layers; ++ring) {
            if (ZI_layers[ring].n_elem == 0) continue;
            arma::mat ZI = ZI_layers[ring];
            arma::mat Zout = ZI;

            for (uint i = 0; i < (ZI.n_rows - 1) / interpol_value; i++) {
                for (uint j = 0; j < ZI.n_cols - 5; j++) {
                    double promedio = 0;
                    double varianza = 0;
                    for (uint k = 0; k < interpol_value; k++) {
                        promedio += ZI((i * interpol_value) + k, j);
                    }
                    promedio /= interpol_value;

                    for (uint l = 0; l < interpol_value; l++) {
                        varianza += std::pow(ZI((i * interpol_value) + l, j) - promedio, 2.0);
                    }
                    varianza = std::sqrt(varianza / interpol_value);

                    if (varianza > max_var) {
                        for (uint m = 0; m < interpol_value; m++) {
                            Zout((i * interpol_value) + m, j) = 0;
                        }
                    }
                }
            }
            ZI_layers[ring] = Zout;
        }
    }

    // Reconstruct 3D point cloud
    PointCloud::Ptr point_cloud(new PointCloud);
    PointCloud::Ptr P_out(new PointCloud);
    point_cloud->is_dense = false;

    int num_pc = 0;
    for (int ring = 0; ring < num_layers; ++ring) {
        if (ZI_layers[ring].n_elem == 0) continue;

        arma::mat ZI = ZI_layers[ring];
        arma::mat ZzI = ZzI_layers[ring];
        point_cloud->width = ZI.n_cols;
        point_cloud->height = ZI.n_rows;
        point_cloud->points.resize(ZI.n_cols * ZI.n_rows);

        for (uint i = 0; i < ZI.n_rows; ++i) {
            for (uint j = 0; j < ZI.n_cols; ++j) {
                if (ZI(i, j) == 0) continue;

                float ang = M_PI - ((2.0 * M_PI * j) / ZI.n_cols);
                float pc_modulo = ZI(i, j);
                float pc_x = std::sqrt(std::pow(pc_modulo, 2) - std::pow(ZzI(i, j), 2)) * std::cos(ang);
                float pc_y = std::sqrt(std::pow(pc_modulo, 2) - std::pow(ZzI(i, j), 2)) * std::sin(ang);

                Eigen::MatrixXf Lidar_matrix(3, 3);
                Eigen::MatrixXf result(3, 1);
                Lidar_matrix << std::cos(ang_x_lidar), 0, std::sin(ang_x_lidar),
                                0, 1, 0,
                                -std::sin(ang_x_lidar), 0, std::cos(ang_x_lidar);
                result << pc_x, pc_y, ZzI(i, j);
                result = Lidar_matrix * result;

                pcl::PointXYZIR pt;
                pt.x = result(0);
                pt.y = result(1);
                pt.z = result(2);
                pt.ring = ring;
                point_cloud->points[num_pc] = pt;
                P_out->push_back(pt);
                num_pc++;
            }
        }
    }

    // Publish point cloud
    P_out->is_dense = true;
    P_out->width = (int)P_out->points.size();
    P_out->height = 1;
    P_out->header.frame_id = "velodyne";
    pc_pub.publish(P_out);

    // Publish range image (using first non-empty layer for simplicity)
    int img_rows = 0, img_cols = 0;
    for (int ring = 0; ring < num_layers; ++ring) {
        if (ZI_layers[ring].n_elem != 0) {
            img_rows = ZI_layers[ring].n_rows;
            img_cols = ZI_layers[ring].n_cols;
            break;
        }
    }
    cv::Mat interdephtImage = cv::Mat::zeros(img_rows, img_cols, cv_bridge::getCvType("mono16"));
    for (int ring = 0; ring < num_layers; ++ring) {
        if (ZI_layers[ring].n_elem == 0) continue;
        for (int i = 0; i < ZI_layers[ring].n_rows; ++i) {
            for (int j = 0; j < ZI_layers[ring].n_cols; ++j) {
                interdephtImage.at<ushort>(i, j) = 1 - (std::pow(2, 16) / (maxlen - minlen)) * (ZI_layers[ring](i, j) - minlen);
            }
        }
    }
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", interdephtImage).toImageMsg();
    imgD_pub.publish(image_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "InterpolatedPointCloud");
    ros::NodeHandle nh;

    // Load parameters
    nh.getParam("/maxlen", maxlen);
    nh.getParam("/minlen", minlen);
    nh.getParam("/pcTopic", pcTopic);
    nh.getParam("/x_resolution", angular_resolution_x);
    nh.getParam("/y_interpolation", interpol_value);
    nh.getParam("/ang_Y_resolution", angular_resolution_y);
    nh.getParam("/ang_ground", ang_x_lidar);
    nh.getParam("/max_var", max_var);
    nh.getParam("/filter_output_pc", f_pc);
    nh.getParam("/num_layers", num_layers);
    nh.getParam("/min_window_size", min_window_size);
    nh.getParam("/max_window_size", max_window_size);

    ros::Subscriber sub = nh.subscribe<PointCloud>(pcTopic, 10, callback);
    pc_pub = nh.advertise<PointCloud>("/pc_interpoled", 10);
    imgD_pub = nh.advertise<sensor_msgs::Image>("/pc2imageInterpol", 10);

    ros::spin();
}
