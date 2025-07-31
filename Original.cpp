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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

typedef std::chrono::high_resolution_clock Clock;
using namespace Eigen;
using namespace sensor_msgs;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

// Publisher
ros::Publisher pc_pub;
ros::Publisher imgD_pub;

float maxlen = 100.0;     // Maximum LiDAR distance
float minlen = 0.01;      // Minimum LiDAR distance
float max_FOV = 3.0;      // Camera max FOV in radians
float min_FOV = 0.4;      // Camera min FOV in radians

// Parameters for converting point cloud to image
float angular_resolution_x = 0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;

float interpol_value = 20.0;
float ang_x_lidar = 0.6 * M_PI / 180.0;
double max_var = 50.0;
bool f_pc = true;

// Topic to subscribe
std::string pcTopic = "/velodyne_points";

// Range image parameters
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

void callback(const PointCloud::ConstPtr& msg_pointCloud)
{
    if (msg_pointCloud == NULL || msg_pointCloud->points.empty()) {
        ROS_WARN("Received empty or null point cloud");
        return;
    }

    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    float max_z = 0, min_z = std::numeric_limits<float>::infinity();
    float max_dis = 0, min_dis = std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < cloud_in->points.size(); ++i) {
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);
        if (distance < minlen || distance > maxlen)
            continue;

        cloud_out->push_back(cloud_in->points[i]);
        if (cloud_in->points[i].z > max_z)
            max_z = cloud_in->points[i].z;
        if (cloud_in->points[i].z < min_z)
            min_z = cloud_in->points[i].z;
        if (distance > max_dis)
            max_dis = distance;
        if (distance < min_dis)
            min_dis = distance;
    }

    // Range image
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    rangeImage->createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                    pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                    sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;

    arma::mat Z;  // Interpolation for range
    arma::mat Zz; // Interpolation for height
    Z.zeros(rows_img, cols_img);
    Zz.zeros(rows_img, cols_img);
    Eigen::MatrixXf ZZei(rows_img, cols_img);

    float max_depth = 0.0;
    float min_depth = -999.0;

    for (int i = 0; i < cols_img; ++i) {
        for (int j = 0; j < rows_img; ++j) {
            float r = rangeImage->getPoint(i, j).range;
            float zz = rangeImage->getPoint(i, j).z;

            Eigen::Vector3f tmp_point;
            rangeImage->calculate3DPoint(float(i), float(j), r, tmp_point);

            if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz)) {
                continue;
            }
            Z.at(j, i) = r;
            Zz.at(j, i) = zz;
            ZZei(j, i) = zz;

            if (r > max_depth)
                max_depth = r;
            if (r < min_depth)
                min_depth = r;
        }
    }

    // Interpolation
    arma::vec X = arma::regspace(1, Z.n_cols);  // Horizontal spacing
    arma::vec Y = arma::regspace(1, Z.n_rows);  // Vertical spacing
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
        // Variance-based filtering
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

    // Reconstruct 3D point cloud
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

            point_cloud->points[num_pc].x = result(0);
            point_cloud->points[num_pc].y = result(1);
            point_cloud->points[num_pc].z = result(2);
            P_out->push_back(point_cloud->points[num_pc]);
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
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "InterpolatedPointCloud");
    ros::NodeHandle nh;

    // Load Parameters
    nh.getParam("/maxlen", maxlen);
    nh.getParam("/minlen", minlen);
    nh.getParam("/pcTopic", pcTopic);
    nh.getParam("/x_resolution", angular_resolution_x);
    nh.getParam("/y_interpolation", interpol_value);
    nh.getParam("/ang_Y_resolution", angular_resolution_y);
    nh.getParam("/ang_ground", ang_x_lidar);
    nh.getParam("/max_var", max_var);
    nh.getParam("/filter_output_pc", f_pc);

    ros::Subscriber sub = nh.subscribe<PointCloud>(pcTopic, 10, callback);
    rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
    pc_pub = nh.advertise<PointCloud>("/pc_interpoled", 10);
    imgD_pub = nh.advertise<sensor_msgs::Image>("/pc2imageInterpol", 10);

    ros::spin();
    return 0;
}
