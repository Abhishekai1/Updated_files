#include <ros/ros.h>
#include <limits>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
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

typedef std::chrono::high_resolution_clock Clock;
using namespace Eigen;
using namespace sensor_msgs;
using namespace std;
using namespace message_filters;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

ros::Publisher pc_pub;
ros::Publisher imgD_pub;

float maxlen = 100.0;
float minlen = 0.01;
float max_FOV = 3.0;
float min_FOV = 0.4;
float angular_resolution_x = 0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;
float interpol_value = 20.0;
float ang_x_lidar = 0.6 * M_PI / 180.0;
double max_var = 50.0;
int num_layers = 16;
bool f_pc = true;

std::string pcTopic = "/velodyne_points";
std::string imgTopic = "/usb_cam/image_raw";

Eigen::MatrixXf tlc(3, 1);
Eigen::MatrixXf rlc(3, 3);
Eigen::MatrixXf camera_matrix(3, 3);

void callback(const PointCloud2ConstPtr& in_pc2, const ImageConstPtr& in_image)
{
    if (!in_pc2 || !in_image) {
        ROS_ERROR("Received null point cloud or image message");
        return;
    }

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*in_pc2, pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

    if (msg_pointCloud->empty()) {
        ROS_WARN("Received empty point cloud");
        return;
    }

    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    if (cloud_in->empty()) {
        ROS_WARN("Point cloud after NaN removal is empty");
        return;
    }

    std::vector<std::vector<pcl::PointXYZI>> layers(num_layers);
    for (size_t i = 0; i < cloud_in->points.size(); ++i) {
        float z = cloud_in->points[i].z;
        float dist = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);
        float angle = atan2(z, dist) * 180.0 / M_PI; // Vertical angle in degrees
        int ring = round((angle + 15.0) / (30.0 / num_layers)); // Map -15° to +15° to 0-15
        if (ring < 0 || ring >= num_layers)
            continue;
        layers[ring].push_back(cloud_in->points[i]);
        cloud_out->push_back(cloud_in->points[i]);
    }

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat image = cv_ptr->image;

    for (int ring = 0; ring < num_layers; ++ring) {
        if (layers[ring].empty())
            continue;

        PointCloud::Ptr layer_cloud(new PointCloud);
        layer_cloud->points.reserve(layers[ring].size()); // Pre-allocate for efficiency
        for (const auto& point : layers[ring]) {
            layer_cloud->points.push_back(point); // Copy points individually
        }
        layer_cloud->width = layer_cloud->points.size();
        layer_cloud->height = 1;
        layer_cloud->is_dense = true;

        // Project points to image
        for (size_t i = 0; i < layer_cloud->points.size(); ++i) {
            Eigen::MatrixXf point(3, 1);
            point << layer_cloud->points[i].x, layer_cloud->points[i].y, layer_cloud->points[i].z;
            Eigen::MatrixXf point_transformed = rlc * point + tlc;
            Eigen::MatrixXf point_camera = camera_matrix * point_transformed;
            if (point_camera(2) == 0.0) {
                ROS_WARN("Zero depth in camera projection, skipping point");
                continue;
            }
            float u = point_camera(0) / point_camera(2);
            float v = point_camera(1) / point_camera(2);

            if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
                cv::circle(image, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), -1);
            }
        }
    }

    if (cloud_out->empty()) {
        ROS_WARN("Output point cloud is empty");
        return;
    }

    // Reconstruct 3D point cloud
    boost::shared_ptr<pcl::RangeImageSpherical> rangeImage(new pcl::RangeImageSpherical);
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    rangeImage->createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                    pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                    sensorPose, pcl::RangeImage::LASER_FRAME, 0.0f, 0.0f, 0);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;
    if (cols_img == 0 || rows_img == 0) {
        ROS_WARN("Range image has invalid dimensions: %d x %d", cols_img, rows_img);
        return;
    }

    arma::mat Z(rows_img, cols_img, arma::fill::zeros);
    arma::mat Zz(rows_img, cols_img, arma::fill::zeros);
    Eigen::MatrixXf ZZei(rows_img, cols_img);

    float max_depth = 0.0;
    float min_depth = -999.0;

    for (int i = 0; i < cols_img; ++i) {
        for (int j = 0; j < rows_img; ++j) {
            float r = rangeImage->getPoint(i, j).range;
            float zz = rangeImage->getPoint(i, j).z;
            if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz))
                continue;
            Z.at(j, i) = r;
            Zz.at(j, i) = zz;
            ZZei(j, i) = zz;
            if (r > max_depth)
                max_depth = r;
            if (r < min_depth)
                min_depth = r;
        }
    }

    arma::vec X = arma::regspace(1, Z.n_cols);
    arma::vec Y = arma::regspace(1, Z.n_rows);
    arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
    arma::vec YI = arma::regspace(Y.min(), 1.0 / interpol_value, Y.max());

    arma::mat ZI, ZzI;
    arma::interp2(X, Y, Z, XI, YI, ZI, "lineal");
    arma::interp2(X, Y, Zz, XI, YI, ZzI, "lineal");

    arma::mat Zout = ZI;
    for (size_t i = 0; i < ZI.n_rows; ++i) {
        for (size_t j = 0; j < ZI.n_cols; ++j) {
            if (ZI(i, j) == 0) {
                if (i + interpol_value < ZI.n_rows)
                    for (size_t k = 1; k <= interpol_value; ++k)
                        Zout(i + k, j) = 0;
                if (i > interpol_value)
                    for (size_t k = 1; k <= interpol_value; ++k)
                        Zout(i - k, j) = 0;
            }
        }
    }
    ZI = Zout;

    if (f_pc) {
        for (size_t i = 0; i < (ZI.n_rows - 1) / interpol_value; ++i) {
            for (size_t j = 0; j < ZI.n_cols - 5; ++j) {
                double promedio = 0;
                for (size_t k = 0; k < interpol_value; ++k)
                    promedio += ZI((i * interpol_value) + k, j);
                promedio /= interpol_value;

                double varianza = 0;
                for (size_t l = 0; l < interpol_value; ++l)
                    varianza += pow((ZI((i * interpol_value) + l, j) - promedio), 2.0);
                varianza = sqrt(varianza / interpol_value);

                if (varianza > max_var)
                    for (size_t m = 0; m < interpol_value; ++m)
                        Zout((i * interpol_value) + m, j) = 0;
            }
        }
        ZI = Zout;
    }

    PointCloud::Ptr point_cloud(new PointCloud);
    PointCloud::Ptr P_out(new PointCloud);
    point_cloud->width = ZI.n_cols;
    point_cloud->height = ZI.n_rows;
    point_cloud->is_dense = false;
    point_cloud->points.resize(point_cloud->width * point_cloud->height);

    int num_pc = 0;
    for (size_t i = 0; i < ZI.n_rows - interpol_value; ++i) {
        for (size_t j = 0; j < ZI.n_cols; ++j) {
            float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));
            if (Zout(i, j) == 0)
                continue;

            float pc_modulo = Zout(i, j);
            float pc_x = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * cos(ang);
            float pc_y = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * sin(ang);

            Eigen::MatrixXf Lidar_matrix(3, 3);
            Eigen::MatrixXf result(3, 1);
            Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                            0, 1, 0,
                            -sin(ang_x_lidar), 0, cos(ang_x_lidar);
            result << pc_x, pc_y, ZzI(i, j);
            result = Lidar_matrix * result;

            pcl::PointXYZI pt;
            pt.x = result(0);
            pt.y = result(1);
            pt.z = result(2);
            pt.intensity = 0.0;
            point_cloud->points[num_pc] = pt;
            P_out->push_back(pt);
            num_pc++;
        }
    }

    P_out->is_dense = true;
    P_out->width = P_out->points.size();
    P_out->height = 1;
    P_out->header.frame_id = "velodyne";
    pc_pub.publish(P_out);

    cv::Mat interdephtImage = cv::Mat::zeros(ZI.n_rows, ZI.n_cols, cv_bridge::getCvType("mono16"));
    for (size_t i = 0; i < ZI.n_cols; ++i) {
        for (size_t j = 0; j < ZI.n_rows; ++j) {
            interdephtImage.at<ushort>(j, i) = (1 - (pow(2, 16) / (maxlen - minlen)) * (ZI(j, i) - minlen));
        }
    }

    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", interdephtImage).toImageMsg();
    imgD_pub.publish(image_msg);

    sensor_msgs::ImagePtr image_msg_color = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    imgD_pub.publish(image_msg_color);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "LidarCameraFusion");
    ros::NodeHandle nh;

    // Load parameters with checks
    if (!nh.getParam("/maxlen", maxlen)) {
        ROS_ERROR("Failed to load maxlen parameter");
        return -1;
    }
    if (!nh.getParam("/minlen", minlen)) {
        ROS_ERROR("Failed to load minlen parameter");
        return -1;
    }
    if (!nh.getParam("/pcTopic", pcTopic)) {
        ROS_ERROR("Failed to load pcTopic parameter");
        return -1;
    }
    if (!nh.getParam("/imgTopic", imgTopic)) {
        ROS_ERROR("Failed to load imgTopic parameter");
        return -1;
    }
    if (!nh.getParam("/x_resolution", angular_resolution_x)) {
        ROS_ERROR("Failed to load x_resolution parameter");
        return -1;
    }
    if (!nh.getParam("/y_interpolation", interpol_value)) {
        ROS_ERROR("Failed to load y_interpolation parameter");
        return -1;
    }
    if (!nh.getParam("/ang_Y_resolution", angular_resolution_y)) {
        ROS_ERROR("Failed to load ang_Y_resolution parameter");
        return -1;
    }
    if (!nh.getParam("/ang_ground", ang_x_lidar)) {
        ROS_ERROR("Failed to load ang_ground parameter");
        return -1;
    }
    if (!nh.getParam("/max_var", max_var)) {
        ROS_ERROR("Failed to load max_var parameter");
        return -1;
    }
    if (!nh.getParam("/filter_output_pc", f_pc)) {
        ROS_ERROR("Failed to load filter_output_pc parameter");
        return -1;
    }
    if (!nh.getParam("/num_layers", num_layers)) {
        ROS_ERROR("Failed to load num_layers parameter");
        return -1;
    }

    // Load transformation matrices with checks
    std::vector<float> tlc_vec, rlc_vec, camera_matrix_vec;
    if (!nh.getParam("/matrix_file/tlc", tlc_vec) || tlc_vec.size() != 3) {
        ROS_ERROR("Failed to load tlc parameter or incorrect size");
        return -1;
    }
    if (!nh.getParam("/matrix_file/rlc", rlc_vec) || rlc_vec.size() != 9) {
        ROS_ERROR("Failed to load rlc parameter or incorrect size");
        return -1;
    }
    if (!nh.getParam("/matrix_file/camera_matrix", camera_matrix_vec) || camera_matrix_vec.size() != 9) {
        ROS_ERROR("Failed to load camera_matrix parameter or incorrect size");
        return -1;
    }

    tlc << tlc_vec[0], tlc_vec[1], tlc_vec[2];
    rlc << rlc_vec[0], rlc_vec[1], rlc_vec[2],
           rlc_vec[3], rlc_vec[4], rlc_vec[5],
           rlc_vec[6], rlc_vec[7], rlc_vec[8];
    camera_matrix << camera_matrix_vec[0], camera_matrix_vec[1], camera_matrix_vec[2],
                     camera_matrix_vec[3], camera_matrix_vec[4], camera_matrix_vec[5],
                     camera_matrix_vec[6], camera_matrix_vec[7], camera_matrix_vec[8];

    message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic, 10);
    message_filters::Subscriber<Image> img_sub(nh, imgTopic, 10);
    typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    pc_pub = nh.advertise<PointCloud>("/points2", 10);
    imgD_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 10);

    ros::spin();
    return 0;
}
