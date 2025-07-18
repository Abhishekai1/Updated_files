// Hybrid and Fixed Version of LiDAR-Camera Fusion Node with Robust Dynamic Interpolation
// Combines modular design from second version + robustness of first

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <armadillo>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

ros::Publisher pc_pub;

float maxlen = 100.0, minlen = 0.01;
int num_layers = 16, base_interp = 6, max_interp = 21;

int dynamic_interp_lines(int layer) {
    return base_interp + layer;
}

arma::mat dynamic_interpolate(const arma::mat& Z, int rows_img, int cols_img) {
    arma::mat ZI(rows_img * max_interp, cols_img, arma::fill::zeros);
    int layer_height = rows_img / num_layers;
    int current_output = 0;

    for (int layer = 0; layer < num_layers; ++layer) {
        int interp_lines = dynamic_interp_lines(layer);
        int start_row = layer * layer_height;
        int end_row = std::min((layer + 1) * layer_height, rows_img);

        arma::mat Z_layer = Z.submat(start_row, 0, end_row - 1, cols_img - 1);

        if (Z_layer.n_rows < 2 || arma::accu(Z_layer != 0) < 4)
            continue;

        arma::vec Y = arma::linspace(1, Z_layer.n_rows, Z_layer.n_rows);
        arma::vec YI = arma::regspace(1.0, 1.0 / interp_lines, Z_layer.n_rows);
        arma::vec X = arma::linspace(1, cols_img, cols_img);
        arma::vec XI = arma::regspace(1.0, 1.0, cols_img);

        arma::mat ZI_layer;
        try {
            arma::interp2(X, Y, Z_layer, XI, YI, ZI_layer, "linear");
        } catch (...) { continue; }

        int rows_to_copy = ZI_layer.n_rows;
        if ((int)(current_output + rows_to_copy) > (int)ZI.n_rows) break;

        ZI.submat(current_output, 0, current_output + rows_to_copy - 1, cols_img - 1) = ZI_layer;
        current_output += rows_to_copy;
    }

    return ZI.rows(0, current_output - 1);
}

void callback(const sensor_msgs::PointCloud2ConstPtr& in_pc2, const sensor_msgs::ImageConstPtr& in_image) {
    cv_bridge::CvImagePtr cv_ptr;
    try { cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8); }
    catch (...) { ROS_ERROR("cv_bridge error"); return; }

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*in_pc2, pcl_pc2);
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

    PointCloud::Ptr filtered(new PointCloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *filtered, indices);
    for (auto& pt : filtered->points) {
        double d = sqrt(pt.x * pt.x + pt.y * pt.y);
        if (d >= minlen && d <= maxlen)
            cloud->points.push_back(pt);
    }

    if (cloud->empty()) return;

    int cols_img = 180;
    int rows_img = 16;
    arma::mat Z(rows_img, cols_img, arma::fill::zeros);

    for (auto& pt : cloud->points) {
        int row = static_cast<int>((pt.z + 15.0) / 2.0);
        int col = static_cast<int>((atan2(pt.y, pt.x) + M_PI) / (2 * M_PI) * cols_img);
        if (row >= 0 && row < rows_img && col >= 0 && col < cols_img) {
            Z(row, col) = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        }
    }

    arma::mat ZI = dynamic_interpolate(Z, rows_img, cols_img);

    PointCloud::Ptr out_pc(new PointCloud);
    for (size_t i = 0; i < ZI.n_rows; ++i) {
        for (size_t j = 0; j < ZI.n_cols; ++j) {
            if (ZI(i, j) > 0) {
                float angle = M_PI - (2.0 * M_PI * j / ZI.n_cols);
                float range = ZI(i, j);
                pcl::PointXYZI p;
                p.x = cos(angle) * range;
                p.y = sin(angle) * range;
                p.z = -15.0f + 2.0f * i;
                p.intensity = range;
                out_pc->points.push_back(p);
            }
        }
    }

    out_pc->header.frame_id = "velodyne";
    pcl_conversions::toPCL(ros::Time::now(), out_pc->header.stamp);
    pc_pub.publish(out_pc);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "robust_dynamic_interpolator");
    ros::NodeHandle nh;

    pc_pub = nh.advertise<PointCloud>("/interpolated_points", 1);

    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub(nh, "/velodyne_points", 1);
    message_filters::Subscriber<sensor_msgs::Image> img_sub(nh, "/camera/color/image_raw", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), pc_sub, img_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();
}
