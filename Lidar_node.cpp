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
#include <algorithm>   // std::max, std::min

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

// Publisher
ros::Publisher pcOnimg_pub;
ros::Publisher pc_pub;

float maxlen = 100.0f;    // lidar max distance
float minlen = 0.01f;     // lidar min distance
float max_FOV = 3.0f;     // camera FOV max (rad)
float min_FOV = 0.4f;     // camera FOV min (rad)

// point-cloud → range-image parameters
float angular_resolution_x = 0.5f;
float angular_resolution_y = 1.0f;   // ↓ from 2.1f for denser vertical sampling
float max_angle_width  = 360.0f;
float max_angle_height = 180.0f;

float max_depth = 100.0f;
float min_depth = 8.0f;
double max_var  = 100.0;  // ↑ tolerate more far-range variance

float interpol_value = 30.0f; // ↑ finer vertical upsampling

bool f_pc = true;

// input topics
std::string imgTopic = "/camera/color/image_raw";
std::string pcTopic  = "/velodyne_points";

// lidar–camera calibration
Eigen::MatrixXf Tlc(3,1); // translation
Eigen::MatrixXf Rlc(3,3); // rotation
Eigen::MatrixXf Mc(3,4);  // camera projection

// range image
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

// ---------- callback ----------
void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2,
              const ImageConstPtr& in_image)
{
  cv_bridge::CvImagePtr cv_ptr, color_pcl;
  try {
    cv_ptr   = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
    color_pcl= cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // sensor_msgs::PointCloud2 → pcl::PointCloud
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*in_pc2, pcl_pc2);
  PointCloud::Ptr msg_pointCloud(new PointCloud);
  pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

  if (!msg_pointCloud) return;

  // Clean NaNs and crop by radius
  PointCloud::Ptr cloud_in(new PointCloud);
  PointCloud::Ptr cloud_out(new PointCloud);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

  for (size_t i = 0; i < cloud_in->points.size(); ++i) {
    const auto& p = cloud_in->points[i];
    double distance = std::sqrt(p.x*p.x + p.y*p.y);
    if (distance < minlen || distance > maxlen) continue;
    cloud_out->push_back(p);
  }

  // Build spherical range image
  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
  rangeImage->pcl::RangeImage::createFromPointCloud(
      *cloud_out,
      pcl::deg2rad(angular_resolution_x),
      pcl::deg2rad(angular_resolution_y),
      pcl::deg2rad(max_angle_width),
      pcl::deg2rad(max_angle_height),
      sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

  const int cols_img = rangeImage->width;
  const int rows_img = rangeImage->height;

  // Depth (range) and height (z) buffers
  arma::mat Z(rows_img, cols_img, arma::fill::zeros);
  arma::mat Zz(rows_img, cols_img, arma::fill::zeros);

  for (int i = 0; i < cols_img; ++i) {
    for (int j = 0; j < rows_img; ++j) {
      const float r  = rangeImage->getPoint(i, j).range;
      const float zz = rangeImage->getPoint(i, j).z;
      if (std::isinf(r) || std::isnan(r) || r < minlen || r > maxlen || std::isnan(zz))
        continue;
      Z.at(j, i)  = r;
      Zz.at(j, i) = zz;
    }
  }

  // Interpolate (vertical upsampling)
  arma::vec X  = arma::regspace(1, Z.n_cols);
  arma::vec Y  = arma::regspace(1, Z.n_rows);
  arma::vec XI = arma::regspace(X.min(), 1.0, X.max());                  // same cols
  arma::vec YI = arma::regspace(Y.min(), 1.0 / interpol_value, Y.max()); // finer rows

  arma::mat ZI, ZzI;
  arma::interp2(X, Y, Z,  XI, YI, ZI,  "lineal");
  arma::interp2(X, Y, Zz, XI, YI, ZzI, "lineal");

  // Gentle zero-wipe around missing bands (preserve far thin structures)
  arma::mat Zout = ZI;
  const int wipe = std::max(1, (int)std::round(interpol_value / 3.0));
  for (arma::uword i = 0; i < ZI.n_rows; ++i) {
    for (arma::uword j = 0; j < ZI.n_cols; ++j) {
      if (ZI(i, j) == 0) {
        // wipe downward
        for (int k = 1; k <= wipe && (i + k) < (int)ZI.n_rows; ++k) Zout(i + k, j) = 0;
        // wipe upward
        for (int k = 1; k <= wipe && (int)i - k >= 0; ++k)           Zout(i - k, j) = 0;
      }
    }
  }
  ZI = Zout;

  // ---------------- Adaptive hole filling (far-range aware) ----------------
  // Parameters (can be exposed later if you want live tuning)
  const double density_threshold = 0.03;  // lower → treat as sparse sooner
  const int    max_window_size   = 15;    // bigger neighborhoods for far range
  const int    min_window_size   = 3;

  // Fill zeros by inverse-distance weighted neighbors with adaptive window
  for (arma::uword i = 1; i + 1 < ZI.n_rows; ++i) {
    for (arma::uword j = 1; j + 1 < ZI.n_cols; ++j) {
      if (ZI(i, j) != 0) continue;

      // local 3×3 valid count
      int valid_neighbors = 0;
      for (int di = -1; di <= 1; ++di)
        for (int dj = -1; dj <= 1; ++dj) {
          arma::sword ni = (arma::sword)i + di;
          arma::sword nj = (arma::sword)j + dj;
          if (ni >= 0 && nj >= 0 && ni < (arma::sword)ZI.n_rows && nj < (arma::sword)ZI.n_cols &&
              ZI(ni, nj) > 0)
            valid_neighbors++;
        }

      const int window = (valid_neighbors < (int)std::ceil(density_threshold * 9.0))
                           ? max_window_size : min_window_size;

      double wsum = 0.0, vsum = 0.0;
      for (int di = -window; di <= window; ++di) {
        for (int dj = -window; dj <= window; ++dj) {
          arma::sword ni = (arma::sword)i + di;
          arma::sword nj = (arma::sword)j + dj;
          if (ni < 0 || nj < 0 || ni >= (arma::sword)ZI.n_rows || nj >= (arma::sword)ZI.n_cols)
            continue;
          const double v = ZI(ni, nj);
          if (v <= 0) continue;

          const double d = std::sqrt((double)(di*di + dj*dj));
          const double w = 1.0 / (d + 1e-6); // IDW
          vsum += w * v;
          wsum += w;
        }
      }
      if (wsum > 0) ZI(i, j) = vsum / wsum;
    }
  }

  // ---------------- Edge-aware preservation ----------------
  arma::mat grad_x(ZI.n_rows, ZI.n_cols, arma::fill::zeros);
  arma::mat grad_y(ZI.n_rows, ZI.n_cols, arma::fill::zeros);
  arma::mat grad_mag(ZI.n_rows, ZI.n_cols, arma::fill::zeros);

  for (arma::uword i = 1; i + 1 < ZI.n_rows; ++i) {
    for (arma::uword j = 1; j + 1 < ZI.n_cols; ++j) {
      if (ZI(i, j) <= 0) continue;
      grad_x(i, j) = (ZI(i, j + 1) - ZI(i, j - 1)) * 0.5;
      grad_y(i, j) = (ZI(i + 1, j) - ZI(i - 1, j)) * 0.5;
      grad_mag(i, j) = std::sqrt(grad_x(i, j)*grad_x(i, j) + grad_y(i, j)*grad_y(i, j));
    }
  }

  const double edge_threshold = 0.05 * arma::max(arma::max(grad_mag)); // more sensitive
  arma::mat Zenhanced = ZI;

  // light bilateral-like smoothing only in non-edge regions
  for (arma::uword i = 1; i + 1 < ZI.n_rows; ++i) {
    for (arma::uword j = 1; j + 1 < ZI.n_cols; ++j) {
      if (ZI(i, j) <= 0) continue;

      if (grad_mag(i, j) <= edge_threshold) {
        // 3×3 smoothing with range-awareness (preserve thin edges)
        double wsum = 0.0, vsum = 0.0;
        const double center = ZI(i, j);
        for (int di = -1; di <= 1; ++di) {
          for (int dj = -1; dj <= 1; ++dj) {
            arma::sword ni = (arma::sword)i + di;
            arma::sword nj = (arma::sword)j + dj;
            if (ni < 0 || nj < 0 || ni >= (arma::sword)ZI.n_rows || nj >= (arma::sword)ZI.n_cols)
              continue;
            const double v = ZI(ni, nj);
            if (v <= 0) continue;

            // spatial weight (IDW) × range weight
            const double d = std::sqrt((double)(di*di + dj*dj));
            const double w_spatial = 1.0 / (d + 1e-6);
            const double dr = std::fabs(v - center);
            const double w_range = 1.0 / (1.0 + dr); // simple robust range term
            const double w = w_spatial * w_range;
            vsum += w * v;
            wsum += w;
          }
        }
        if (wsum > 0) Zenhanced(i, j) = vsum / wsum;
      } else {
        // damp strong-gradient oversmoothing
        double weight = std::max(0.0, 1.0 - grad_mag(i, j) / (edge_threshold + 1e-6));
        Zenhanced(i, j) = ZI(i, j) * weight + Zenhanced(i, j) * (1.0 - weight);
      }
    }
  }
    
  # Integrated Canny Edge Detection in Interpolation Phase
// Paste below line:
//     ZI = Zenhanced;
// Add Canny processing for edge preservation after bilateral smoothing.

cv::Mat ZI_cv(ZI.n_rows, ZI.n_cols, CV_32FC1);
for (arma::uword i = 0; i < ZI.n_rows; ++i)
  for (arma::uword j = 0; j < ZI.n_cols; ++j)
    ZI_cv.at<float>(i, j) = static_cast<float>(ZI(i, j));

cv::Mat ZI_8U, edges;
double minVal, maxVal;
cv::minMaxLoc(ZI_cv, &minVal, &maxVal);
ZI_cv.convertTo(ZI_8U, CV_8U, 255.0 / (maxVal - minVal + 1e-6));

// Apply Gaussian blur before Canny
cv::GaussianBlur(ZI_8U, ZI_8U, cv::Size(5,5), 1.0);
cv::Canny(ZI_8U, edges, 50, 150);

// Mask out high edge regions (preserve edges)
for (arma::uword i = 1; i + 1 < ZI.n_rows; ++i) {
  for (arma::uword j = 1; j + 1 < ZI.n_cols; ++j) {
    if (edges.at<uchar>(i, j) > 0) {
      Zenhanced(i, j) = ZI(i, j);  // don't smooth at strong edge
    }
  }
}

  ZI = Zenhanced;

  // ---------------- Variance-based de-flicker with far-range leniency ----------------
  if (f_pc) {
    Zout = ZI;
    for (arma::uword i = 0; i < (ZI.n_rows > 0 ? (ZI.n_rows - 1) / (arma::uword)interpol_value : 0); ++i) {
      for (arma::uword j = 0; j + 5 < ZI.n_cols; ++j) {
        double sum = 0.0;
        for (arma::uword k = 0; k < (arma::uword)interpol_value; ++k) sum += ZI(i*interpol_value + k, j);
        const double mean = sum / std::max< double >(1.0, (double)interpol_value);

        double var = 0.0;
        for (arma::uword k = 0; k < (arma::uword)interpol_value; ++k) {
          const double dv = ZI(i*interpol_value + k, j) - mean;
          var += dv * dv;
        }
        var = std::sqrt(var / std::max< double >(1.0, (double)interpol_value));

        // farther means noisier → be lenient
        double var_adj = var;
        if (mean > maxlen * 0.5) var_adj *= 1.5;

        if (var_adj > max_var) {
          for (arma::uword m = 0; m < (arma::uword)interpol_value; ++m)
            Zout(i*interpol_value + m, j) = 0;
        }
      }
    }
    ZI = Zout;
  }

  // ---------------- Range image → 3D cloud ----------------
  PointCloud::Ptr point_cloud(new PointCloud);
  PointCloud::Ptr cloud(new PointCloud);
  point_cloud->width  = ZI.n_cols;
  point_cloud->height = ZI.n_rows;
  point_cloud->is_dense = false;
  point_cloud->points.resize(point_cloud->width * point_cloud->height);

  int num_pc = 0;
  for (arma::uword i = 0; i + (arma::uword)interpol_value < ZI.n_rows; ++i) {
    for (arma::uword j = 0; j < ZI.n_cols; ++j) {
      const float ang = M_PI - ((2.0f * M_PI * (float)j) / (float)ZI.n_cols);
      if (ang < (min_FOV - M_PI/2.0f) || ang > (max_FOV - M_PI/2.0f)) continue;

      if (ZI(i, j) == 0) continue;

      const float pc_mod = (float)ZI(i, j);
      const float zval   = (float)ZzI(i, j);
      const float planar = std::sqrt(std::max(0.0f, pc_mod*pc_mod - zval*zval));

      float pc_x = planar * std::cos(ang);
      float pc_y = planar * std::sin(ang);

      const float ang_x_lidar = 0.6f * M_PI / 180.0f; // small pitch correction
      Eigen::Matrix3f Lidar_matrix;
      Lidar_matrix <<  std::cos(ang_x_lidar), 0,  std::sin(ang_x_lidar),
                        0,                    1,  0,
                       -std::sin(ang_x_lidar),0,  std::cos(ang_x_lidar);

      Eigen::Vector3f v(pc_x, pc_y, zval);
      Eigen::Vector3f r = Lidar_matrix * v;

      point_cloud->points[num_pc].x = r(0);
      point_cloud->points[num_pc].y = r(1);
      point_cloud->points[num_pc].z = r(2);
      cloud->push_back(point_cloud->points[num_pc]);
      ++num_pc;
    }
  }

  // Optional: outlier removal or voxel grid (kept disabled to preserve density)
  PointCloud::Ptr P_out(new PointCloud);
  P_out = cloud;

  // ---------------- Project to image & colorize ----------------
  Eigen::MatrixXf RTlc(4,4);
  RTlc << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
          Rlc(1), Rlc(4), Rlc(7), Tlc(1),
          Rlc(2), Rlc(5), Rlc(8), Tlc(2),
          0,      0,      0,      1;

  const int size_inter_Lidar = (int)P_out->points.size();
  Eigen::MatrixXf Lidar_cam(3,1);
  Eigen::MatrixXf pc_matrix(4,1);

  const unsigned int cols = in_image->width;
  const unsigned int rows = in_image->height;

  uint px_data = 0, py_data = 0;
  pcl::PointXYZRGB point;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color(new pcl::PointCloud<pcl::PointXYZRGB>);

  for (int i = 0; i < size_inter_Lidar; ++i) {
    pc_matrix(0,0) = -P_out->points[i].y;
    pc_matrix(1,0) = -P_out->points[i].z;
    pc_matrix(2,0) =  P_out->points[i].x;
    pc_matrix(3,0) = 1.0;

    Lidar_cam = Mc * (RTlc * pc_matrix);
    px_data = (int)(Lidar_cam(0,0) / Lidar_cam(2,0));
    py_data = (int)(Lidar_cam(1,0) / Lidar_cam(2,0));
    if (px_data < 0 || (unsigned)px_data >= cols || py_data < 0 || (unsigned)py_data >= rows)
      continue;

    const float dist_xy = std::sqrt(P_out->points[i].x * P_out->points[i].x +
                                    P_out->points[i].y * P_out->points[i].y);
    int color_dis_x = (int)(255.0f * ((dist_xy / maxlen) * 1.2f));
    if (color_dis_x > 255) color_dis_x = 255;
    int color_dis_z = (int)(255.0f * ((P_out->points[i].z) / 10.0f));
    if (color_dis_z > 255) color_dis_z = 255;

    cv::Vec3b& color = color_pcl->image.at<cv::Vec3b>(py_data, px_data);

    point.x = P_out->points[i].x;
    point.y = P_out->points[i].y;
    point.z = P_out->points[i].z;
    point.r = (int)color[2];
    point.g = (int)color[1];
    point.b = (int)color[0];
    pc_color->points.push_back(point);

    cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1,
               CV_RGB(255 - color_dis_x, (int)(color_dis_z), color_dis_x), cv::FILLED);
  }

  pc_color->is_dense = true;
  pc_color->width  = (int)pc_color->points.size();
  pc_color->height = 1;
  pc_color->header.frame_id = "velodyne";

  pcOnimg_pub.publish(cv_ptr->toImageMsg());
  pc_pub.publish(pc_color);
}

// ---------- main ----------
int main(int argc, char** argv)
{
  ros::init(argc, argv, "pontCloudOntImage");
  ros::NodeHandle nh;

  // Load Parameters (same names as before; you can override at runtime)
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
  Tlc <<  (double)param[0],
          (double)param[1],
          (double)param[2];

  nh.getParam("/matrix_file/rlc", param);
  Rlc <<  (double)param[0], (double)param[1], (double)param[2],
          (double)param[3], (double)param[4], (double)param[5],
          (double)param[6], (double)param[7], (double)param[8];

  nh.getParam("/matrix_file/camera_matrix", param);
  Mc  << (double)param[0],  (double)param[1],  (double)param[2],  (double)param[3],
         (double)param[4],  (double)param[5],  (double)param[6],  (double)param[7],
         (double)param[8],  (double)param[9],  (double)param[10], (double)param[11];

  message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic, 1);
  message_filters::Subscriber<Image>      img_sub(nh, imgTopic, 1);

  typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);
  rangeImage   = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
  pc_pub       = nh.advertise<PointCloud>("/points2", 1);

  ros::spin();
}
