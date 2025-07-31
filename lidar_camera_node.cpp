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
#include <velodyne_pointcloud/point_types.h> // Added for PointXYZIR
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <armadillo>
#include <Eigen/Dense>
#include <math.h>
#include <chrono>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

// Use PointXYZIR for ring information
typedef pcl::PointCloud<pcl::PointXYZIR> PointCloud;

// Publishers
ros::Publisher pcOnimg_pub;
ros::Publisher pc_pub;

float maxlen = 100.0;     // Maximum LiDAR distance
float minlen = 0.01;      // Minimum LiDAR distance
float max_FOV = 3.0;      // Camera max FOV in radians
float min_FOV = 0.4;      // Camera min FOV in radians

// Parameters for point cloud to image conversion
float angular_resolution_x = 0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth = 100.0;
float min_depth = 8.0;
double max_var = 50.0;
float interpol_value = 20.0;
bool f_pc = true;
int num_layers = 16; // Number of LiDAR layers (e.g., 16 for VLP-16)
int min_window_size = 3; // Minimum interpolation kernel size
int max_window_size = 9; // Maximum interpolation kernel size

// Input topics
std::string imgTopic = "/camera/color/image_raw";
std::string pcTopic = "/velodyne_points";

// Calibration matrices
Eigen::MatrixXf Tlc(3, 1); // Translation matrix lidar-camera
Eigen::MatrixXf Rlc(3, 3); // Rotation matrix lidar-camera
Eigen::MatrixXf Mc(3, 4);  // Camera calibration matrix

// Range image parameters
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

// Function to determine kernel size based on layer
int getKernelSize(int layer) {
    return min_window_size + (layer * (max_window_size - min_window_size) / (num_layers - 1));
}

void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2, const ImageConstPtr& in_image)
{
    // Convert image
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
    pcl_conversions::toPCL(*in_pc2, pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

    if (msg_pointCloud == NULL || msg_pointCloud->points.empty()) {
        ROS_WARN("Received empty or null point cloud");
        return;
    }

    // Filter point cloud
    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    // Split points into layers
    std::vector<std::vector<pcl::PointXYZIR>> layers(num_layers);
    for (const auto& pt : cloud_in->points) {
        double distance = sqrt(pt.x * pt.x + pt.y * pt.y);
        if (distance < minlen || distance > maxlen) continue;

        int ring = pt.ring;
        if (ring >= 0 && ring < num_layers) {
            layers[ring].push_back(pt);
            cloud_out->push_back(pt);
        }
    }

    // Create range images per layer
    std::vector<boost::shared_ptr<pcl::RangeImageSpherical>> rangeImages(num_layers);
    std::vector<arma::mat> Z_layers(num_layers);
    std::vector<arma::mat> Zz_layers(num_layers);

    for (int ring = 0; ring < num_layers; ++ring) {
        if (layers[ring].empty()) {
            ROS_DEBUG("Layer %d is empty, skipping", ring);
            continue;
        }

        rangeImages[ring] = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
        Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
        PointCloud::Ptr layer_cloud(new PointCloud);
        *layer_cloud = PointCloud(layers[ring].begin(), layers[ring].end());

        rangeImages[ring]->createFromPointCloud(*layer_cloud, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                                pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                                sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

        int cols_img = rangeImages[ring]->width;
        int rows_img = 1;
        Z_layers[ring].zeros(rows_img, cols_img);
        Zz_layers[ring].zeros(rows_img, cols_img);

        for (int i = 0; i < cols_img; ++i) {
            float r = rangeImages[ring]->getPoint(i, 0).range;
            float zz = rangeImages[ring]->getPoint(i, 0).z;
            if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz)) continue;
            Z_layers[ring].at(0, i) = r;
            Zz_layers[ring].at(0, i) = zz;
        }
    }

    // Dynamic interpolation per layer
    std::vector<arma::mat> ZI_layers(num_layers);
    std::vector<arma::mat> ZzI_layers(num_layers);

    for (int ring = 0; ring < num_layers; ++ring) {
        if (Z_layers[ring].n_elem == 0) {
            ROS_DEBUG("No data in layer %d, skipping interpolation", ring);
            continue;
        }

        arma::vec X = arma::regspace(1, Z_layers[ring].n_cols);
        arma::vec Y = arma::regspace(1, 1);
        arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
        arma::vec YI = arma::regspace(Y.min(), 1.0 / interpol_value, Y.max());

        arma::mat ZI, ZzI;
        arma::interp2(X, Y, Z_layers[ring], XI, YI, ZI, "lineal");
        arma::interp2(X, Y, Zz_layers[ring], XI, YI, ZzI, "lineal");

        arma::mat Zout = ZI;
        int window_size = getKernelSize(ring);
        // double density_threshold = 0.05; // Unused variable, commented out

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
                            if (ni >= 0 && nj >= 0 && ni < (int)ZI.n_rows && nj < (int)ZI.n_cols && ZI(ni, nj) > 0) {
                                valid_neighbors++;
                            }
                        }
                    }

                    for (int di = -window_size; di <= window_size; ++di) {
                        for (int dj = -window_size; dj <= window_size; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            if (ni >= 0 && nj >= 0 && ni < (int)ZI.n_rows && nj < (int)ZI.n_cols && ZI(ni, nj) > 0) {
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
    PointCloud::Ptr cloud(new PointCloud);
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
                float ang = M_PI - ((2.0 * M_PI * j) / ZI.n_cols);
                if (ang < min_FOV - M_PI / 2.0 || ang > max_FOV - M_PI / 2.0) continue;
                if (ZI(i, j) == 0) continue;

                float pc_modulo = ZI(i, j);
                float pc_x = std::sqrt(std::pow(pc_modulo, 2) - std::pow(ZzI(i, j), 2)) * std::cos(ang);
                float pc_y = std::sqrt(std::pow(pc_modulo, 2) - std::pow(ZzI(i, j), 2)) * std::sin(ang);

                float ang_x_lidar = 0.6 * M_PI / 180.0;
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
                cloud->push_back(pt);
                num_pc++;
            }
        }
    }

    PointCloud::Ptr P_out(new PointCloud);
    P_out = cloud;

    // Camera projection
    Eigen::MatrixXf RTlc(4, 4);
    RTlc << Rlc(0, 0), Rlc(0, 1), Rlc(0, 2), Tlc(0),
            Rlc(1, 0), Rlc(1, 1), Rlc(1, 2), Tlc(1),
            Rlc(2, 0), Rlc(2, 1), Rlc(2, 2), Tlc(2),
            0, 0, 0, 1;

    int size_inter_Lidar = (int)P_out->points.size();
    Eigen::MatrixXf Lidar_cam(3, 1);
    Eigen::MatrixXf pc_matrix(4, 1);

    unsigned int cols = in_image->width;
    unsigned int rows = in_image->height;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (int i = 0; i < size_inter_Lidar; i++) {
        pc_matrix(0, 0) = -P_out->points[i].y;
        pc_matrix(1, 0) = -P_out->points[i].z;
        pc_matrix(2, 0) = P_out->points[i].x;
        pc_matrix(3, 0) = 1.0;

        Lidar_cam = Mc * (RTlc * pc_matrix);
        uint px_data = (int)(Lidar_cam(0, 0) / Lidar_cam(2, 0));
        uint py_data = (int)(Lidar_cam(1, 0) / Lidar_cam(2, 0));

        if (px_data < 0 || px_data >= cols || py_data < 0 || py_data >= rows) continue;

        int color_dis_x = (int)(255 * (P_out->points[i].x / maxlen));
        int color_dis_z = (int)(255 * (P_out->points[i].x / 10.0));
        if (color_dis_z > 255) color_dis_z = 255;

        cv::Vec3b &color = color_pcl->image.at<cv::Vec3b>(py_data, px_data);

        pcl::PointXYZRGB point;
        point.x = P_out->points[i].x;
        point.y = P_out->points[i].y;
        point.z = P_out->points[i].z;
        point.r = (int)color[2];
        point.g = (int)color[1];
        point.b = (int)color[0];
        pc_color->points.push_back(point);

        cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1, CV_RGB(255 - color_dis_x, color_dis_z, color_dis_x), cv::FILLED);
    }

    if (!pc_color->points.empty()) {
        pc_color->is_dense = true;
        pc_color->width = (int)pc_color->points.size();
        pc_color->height = 1;
        pc_color->header.frame_id = "velodyne";
        pc_pub.publish(pc_color);
    } else {
        ROS_WARN("No colored points to publish");
    }

    pcOnimg_pub.publish(cv_ptr->toImageMsg());
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pontCloudOntImage");
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
    nh.getParam("/y_interpolation", interpol_value);
    nh.getParam("/ang_Y_resolution", angular_resolution_y);
    nh.getParam("/num_layers", num_layers);
    nh.getParam("/min_window_size", min_window_size);
    nh.getParam("/max_window_size", max_window_size);

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
    pc_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/points2", 1);

    ros::spin();
    return 0;
}
