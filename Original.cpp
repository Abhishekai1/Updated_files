#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_transport/image_transport.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <armadillo>
#include <Eigen/Dense>
#include <math.h>
#include <chrono>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

// Publishers
ros::Publisher pc_pub;
ros::Publisher imgD_pub;

float maxlen = 100.0;
float minlen = 0.01;

// Assume other parameters and functions similar to interpolated_lidar.cpp
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

    // Example processing (replace with actual orginal.cpp logic)
    arma::mat ZI(cloud_in->height, cloud_in->width);
    for (size_t i = 0; i < ZI.n_cols; ++i) {
        for (size_t j = 0; j < ZI.n_rows; ++j) {
            // Example computation
            ZI(j, i) = cloud_in->points[i * ZI.n_rows + j].x;
        }
    }

    // Publish point cloud
    if (!cloud_out->points.empty()) {
        cloud_out->is_dense = true;
        cloud_out->width = cloud_out->points.size();
        cloud_out->height = 1;
        cloud_out->header.frame_id = "velodyne";
        pc_pub.publish(cloud_out);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "OrginalPointCloud");
    ros::NodeHandle nh;

    std::string pcTopic = "/velodyne_points";
    nh.getParam("/maxlen", maxlen);
    nh.getParam("/minlen", minlen);
    nh.getParam("/pcTopic", pcTopic);

    ros::Subscriber sub = nh.subscribe<PointCloud>(pcTopic, 10, callback);
    pc_pub = nh.advertise<PointCloud>("/orginal_points", 10);
    imgD_pub = nh.advertise<sensor_msgs::Image>("/orginal_image", 10);

    ros::spin();
    return 0;
}
