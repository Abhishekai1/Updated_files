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

float maxlen =100.0;       //maxima distancia del lidar
float minlen = 0.01;     //minima distancia del lidar
float max_FOV = 3.0;    // en radianes angulo maximo de vista de la camara
float min_FOV = 0.4;    // en radianes angulo minimo de vista de la camara

/// parametros para convertir nube de puntos en imagen
float angular_resolution_x =0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width= 360.0f;
float max_angle_height = 180.0f;
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth =100.0;
float min_depth = 8.0;
double max_var = 50.0; 

float interpol_value = 20.0;

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

///////////////////////////////////////callback

void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2 , const ImageConstPtr& in_image)
{
    cv_bridge::CvImagePtr cv_ptr , color_pcl;
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
  pcl_conversions::toPCL(*in_pc2,pcl_pc2);
  PointCloud::Ptr msg_pointCloud(new PointCloud);
  pcl::fromPCLPointCloud2(pcl_pc2,*msg_pointCloud);

  ////// filter point cloud 
  if (msg_pointCloud == NULL) return;

  PointCloud::Ptr cloud_in (new PointCloud);
  PointCloud::Ptr cloud_out (new PointCloud);

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);
  
  for (int i = 0; i < (int) cloud_in->points.size(); i++)
  {
      double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);     
      if(distance<minlen || distance>maxlen)
       continue;        
      cloud_out->push_back(cloud_in->points[i]);     
  }  

  //                                                  point cloud to image 
  //============================================================================================================
  //============================================================================================================

  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
  rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                       pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                       sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

  int cols_img = rangeImage->width;
  int rows_img = rangeImage->height;

  arma::mat Z;  // interpolation de la imagen
  arma::mat Zz; // interpolation de las alturas de la imagen

  Z.zeros(rows_img,cols_img);         
  Zz.zeros(rows_img,cols_img);       

  Eigen::MatrixXf ZZei (rows_img,cols_img);
 
  for (int i=0; i< cols_img; ++i)
      for (int j=0; j<rows_img ; ++j)
      {
        float r =  rangeImage->getPoint(i, j).range;     
        float zz = rangeImage->getPoint(i, j).z; 

        if(std::isinf(r) || r<minlen || r>maxlen || std::isnan(zz)){
            continue;
        }             
        Z.at(j,i) = r;   
        Zz.at(j,i) = zz;
      }

  ////////////////////////////////////////////// Enhanced interpolation section
  //============================================================================================================
  
  // VLP-16 specific parameters
  const int VLP16_LAYERS = 16;
  const int BASE_INTERPOLATION = 6;  // Starting interpolation lines for first layer
  const int MAX_INTERPOLATION = 21;  // Maximum interpolation lines for last layer

  // Calculate dynamic interpolation value for each layer
  auto calculateDynamicInterpolation = [](int layer_index, int total_layers) -> float {
      // Linear increase from BASE_INTERPOLATION to MAX_INTERPOLATION
      float interpolation_step = (float)(MAX_INTERPOLATION - BASE_INTERPOLATION) / (total_layers - 1);
      return BASE_INTERPOLATION + (interpolation_step * layer_index);
  };

  // Create dynamic interpolation matrix
  arma::mat ZI_dynamic;
  arma::mat ZzI_dynamic;

  // Calculate total rows needed for dynamic interpolation
  int total_interpolated_rows = 0;
  for (int layer = 0; layer < VLP16_LAYERS; layer++) {
      float layer_interpolation = calculateDynamicInterpolation(layer, VLP16_LAYERS);
      if (layer < Z.n_rows) {
          total_interpolated_rows += (int)layer_interpolation;
      }
  }

  // Initialize dynamic interpolation matrices
  ZI_dynamic.zeros(total_interpolated_rows, Z.n_cols);
  ZzI_dynamic.zeros(total_interpolated_rows, Z.n_cols);

  // Process each layer with dynamic interpolation
  int output_row_index = 0;
  int layer_height = Z.n_rows / VLP16_LAYERS;  // Approximate rows per layer

  for (int layer = 0; layer < VLP16_LAYERS && layer * layer_height < Z.n_rows; layer++) {
      float current_interpolation = calculateDynamicInterpolation(layer, VLP16_LAYERS);
      
      // Define layer boundaries
      int layer_start = layer * layer_height;
      int layer_end = std::min((layer + 1) * layer_height, (int)Z.n_rows);
      
      // Extract current layer data
      arma::mat layer_Z = Z.submat(layer_start, 0, layer_end - 1, Z.n_cols - 1);
      arma::mat layer_Zz = Zz.submat(layer_start, 0, layer_end - 1, Zz.n_cols - 1);
      
      // Create interpolation vectors for this layer
arma::vec X_layer = arma::regspace(1, layer_Z.n_cols);
arma::vec Y_layer = arma::regspace(1, layer_Z.n_rows);
arma::vec unique_Y = arma::unique(Y_layer);

if (layer_Z.n_rows < 2 || unique_Y.n_elem < 2) {
    ROS_WARN("Skipping layer %d: Y vector doesn't have enough unique elements.", layer);
    continue;
}



      
      // Dynamic interpolation spacing
      arma::vec YI_layer = arma::regspace(Y_layer.min(), 1.0/current_interpolation, Y_layer.max());
      arma::vec XI_layer = arma::regspace(X_layer.min(), 1.0, X_layer.max());
      
      // Perform interpolation for this layer
      arma::mat ZI_layer, ZzI_layer;
      
      try {
          arma::interp2(X_layer, Y_layer, layer_Z, XI_layer, YI_layer, ZI_layer, "linear");
          arma::interp2(X_layer, Y_layer, layer_Zz, XI_layer, YI_layer, ZzI_layer, "linear");
          
          // Insert interpolated layer into dynamic result
          int layer_rows = ZI_layer.n_rows;
          int available_rows = std::min(layer_rows, total_interpolated_rows - output_row_index);
          
          if (available_rows > 0) {
              ZI_dynamic.submat(output_row_index, 0, output_row_index + available_rows - 1, ZI_layer.n_cols - 1) = 
                  ZI_layer.submat(0, 0, available_rows - 1, ZI_layer.n_cols - 1);
              
              ZzI_dynamic.submat(output_row_index, 0, output_row_index + available_rows - 1, ZzI_layer.n_cols - 1) = 
                  ZzI_layer.submat(0, 0, available_rows - 1, ZzI_layer.n_cols - 1);
              
              output_row_index += available_rows;
          }
          
          // Debug output for layer information
          ROS_INFO("Layer %d: interpolation=%.1f, rows=%d, output_index=%d", 
                   layer, current_interpolation, layer_rows, output_row_index);
          
      } catch (const std::exception& e) {
          ROS_WARN("Interpolation failed for layer %d: %s", layer, e.what());
          continue;
      }
  }

  // Replace original interpolation results with dynamic results
  arma::mat ZI = ZI_dynamic;
  arma::mat ZzI = ZzI_dynamic;

  // Enhanced zero handling with layer-aware interpolation
  arma::mat Zout = ZI;

  // Apply layer-aware zero filtering
  output_row_index = 0;
  for (int layer = 0; layer < VLP16_LAYERS; layer++) {
      float current_interpolation = calculateDynamicInterpolation(layer, VLP16_LAYERS);
      int layer_rows = (int)current_interpolation * (layer_height > 0 ? layer_height : 1);
      
      // Ensure we don't exceed matrix bounds
      int actual_layer_rows = std::min(layer_rows, (int)ZI.n_rows - output_row_index);
      if (actual_layer_rows <= 0) break;
      
      // Process zeros in this layer
      for (int i = output_row_index; i < output_row_index + actual_layer_rows; i++) {
          for (uint j = 0; j < ZI.n_cols; j++) {
              if (ZI(i, j) == 0) {
                  int interpolation_range = (int)(current_interpolation * 0.5); // Adaptive range
                  
                  // Forward zero propagation
                  if (i + interpolation_range < ZI.n_rows) {
                      for (int k = 1; k <= interpolation_range; k++) {
                          if (i + k < ZI.n_rows) {
                              Zout(i + k, j) = 0;
                          }
                      }
                  }
                  
                  // Backward zero propagation
                  if (i > interpolation_range) {
                      for (int k = 1; k <= interpolation_range; k++) {
                          if (i - k >= 0) {
                              Zout(i - k, j) = 0;
                          }
                      }
                  }
              }
          }
      }
      
      output_row_index += actual_layer_rows;
  }

  ZI = Zout;

  // Enhanced adaptive interpolation with layer awareness
  double base_density_threshold = 0.05;
  int base_max_window = 9;
  int base_min_window = 3;

  output_row_index = 0;
  for (int layer = 0; layer < VLP16_LAYERS; layer++) {
      float current_interpolation = calculateDynamicInterpolation(layer, VLP16_LAYERS);
      int layer_rows = (int)current_interpolation * (layer_height > 0 ? layer_height : 1);
      
      // Ensure we don't exceed matrix bounds
      int actual_layer_rows = std::min(layer_rows, (int)ZI.n_rows - output_row_index);
      if (actual_layer_rows <= 0) break;
      
      // Adaptive parameters based on layer interpolation density
      double layer_density_threshold = base_density_threshold * (current_interpolation / BASE_INTERPOLATION);
      int layer_max_window = base_max_window + (int)((current_interpolation - BASE_INTERPOLATION) / 2.0);
      int layer_min_window = base_min_window;
      
      // Process missing data in this layer
      for (int i = output_row_index + 1; i < output_row_index + actual_layer_rows - 1; i++) {
          for (uint j = 1; j < ZI.n_cols - 1; j++) {
              if (ZI(i, j) == 0) {
                  double weighted_sum = 0.0;
                  double weight_total = 0.0;
                  
                  // Calculate local density
                  int valid_neighbors = 0;
                  for (int di = -1; di <= 1; di++) {
                      for (int dj = -1; dj <= 1; dj++) {
                          int ni = i + di;
                          int nj = j + dj;
                          
                          if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                              valid_neighbors++;
                          }
                      }
                  }
                  
                  // Determine window size based on layer density
                  int window_size = (valid_neighbors < layer_density_threshold * 9) ? layer_max_window : layer_min_window;
                  
                  // Interpolate using determined window size
                  for (int di = -window_size; di <= window_size; di++) {
                      for (int dj = -window_size; dj <= window_size; dj++) {
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
      
      output_row_index += actual_layer_rows;
  }

  // Enhanced edge preservation with layer-specific thresholds
  arma::mat Zenhanced = ZI;
  arma::mat grad_x = arma::zeros(ZI.n_rows, ZI.n_cols);
  arma::mat grad_y = arma::zeros(ZI.n_rows, ZI.n_cols);
  arma::mat grad_mag = arma::zeros(ZI.n_rows, ZI.n_cols);

  // Calculate gradients with layer awareness
  for (uint i = 1; i < ZI.n_rows - 1; i++) {
      for (uint j = 1; j < ZI.n_cols - 1; j++) {
          if (ZI(i, j) > 0) {
              grad_x(i, j) = (ZI(i, j + 1) - ZI(i, j - 1)) * 0.5;
              grad_y(i, j) = (ZI(i + 1, j) - ZI(i - 1, j)) * 0.5;
              grad_mag(i, j) = std::sqrt(grad_x(i, j) * grad_x(i, j) + grad_y(i, j) * grad_y(i, j));
          }
      }
  }

  // Apply layer-specific edge preservation
  double base_edge_threshold = 0.1 * arma::max(arma::max(grad_mag));
  output_row_index = 0;

  for (int layer = 0; layer < VLP16_LAYERS; layer++) {
      float current_interpolation = calculateDynamicInterpolation(layer, VLP16_LAYERS);
      int layer_rows = (int)current_interpolation * (layer_height > 0 ? layer_height : 1);
      
      // Ensure we don't exceed matrix bounds
      int actual_layer_rows = std::min(layer_rows, (int)ZI.n_rows - output_row_index);
      if (actual_layer_rows <= 0) break;
      
      // Layer-specific edge threshold
      double layer_edge_threshold = base_edge_threshold * (current_interpolation / BASE_INTERPOLATION);
      
      for (int i = output_row_index + 1; i < output_row_index + actual_layer_rows - 1; i++) {
          for (uint j = 1; j < ZI.n_cols - 1; j++) {
              if (grad_mag(i, j) > layer_edge_threshold) {
                  double weight = std::max(0.0, 1.0 - grad_mag(i, j) / layer_edge_threshold);
                  Zenhanced(i, j) = ZI(i, j) * weight + Zenhanced(i, j) * (1 - weight);
              }
          }
      }
      
      output_row_index += actual_layer_rows;
  }

  // Replace with enhanced result
  ZI = Zenhanced;

  // Update variance filtering to work with dynamic interpolation
  if (f_pc) {
      arma::mat Zout_var = ZI;
      output_row_index = 0;
      
      for (int layer = 0; layer < VLP16_LAYERS; layer++) {
          float current_interpolation = calculateDynamicInterpolation(layer, VLP16_LAYERS);
          int layer_rows = (int)current_interpolation * (layer_height > 0 ? layer_height : 1);
          
          // Ensure we don't exceed matrix bounds
          int actual_layer_rows = std::min(layer_rows, (int)ZI.n_rows - output_row_index);
          if (actual_layer_rows <= 0) break;
          
          // Process variance filtering for this layer
          for (int i = output_row_index; i < output_row_index + actual_layer_rows - (int)current_interpolation; i += 1) {
              for (uint j = 0; j < ZI.n_cols - 5; j += 1) {
                  double promedio = 0;
                  double varianza = 0;
                  
                  // Calculate average using current layer's interpolation
                  for (int k = 0; k < (int)current_interpolation; k += 1) {
                      if (i + k < ZI.n_rows) {
                          promedio += ZI(i + k, j);
                      }
                  }
                  promedio /= current_interpolation;
                  
                  // Calculate variance
                  for (int l = 0; l < (int)current_interpolation; l++) {
                      if (i + l < ZI.n_rows) {
                          varianza += pow((ZI(i + l, j) - promedio), 2.0);
                      }
                  }
                  
                  // Apply variance threshold (adjusted for layer density)
                  double layer_max_var = max_var * (current_interpolation / BASE_INTERPOLATION);
                  if (varianza > layer_max_var) {
                      for (int m = 0; m < (int)current_interpolation; m++) {
                          if (i + m < ZI.n_rows) {
                              Zout_var(i + m, j) = 0;
                          }
                      }
                  }
              }
          }
          
          output_row_index += actual_layer_rows;
      }
      
      ZI = Zout_var;
  }

  // Dynamic interpolation complete
  ROS_INFO("Dynamic interpolation complete. Total rows: %d", (int)ZI.n_rows);

  //===========================================fin filtrado por imagen=================================================

  // reconstruccion de imagen a nube 3D
  //============================================================================================================

  PointCloud::Ptr point_cloud (new PointCloud);
  PointCloud::Ptr cloud (new PointCloud);
  point_cloud->width = ZI.n_cols; 
  point_cloud->height = ZI.n_rows;
  point_cloud->is_dense = false;
  point_cloud->points.resize (point_cloud->width * point_cloud->height);

  ///////// imagen de rango a nube de puntos  
  int num_pc = 0; 
  for (uint i=0; i< ZI.n_rows - interpol_value; i+=1)
   {       
      for (uint j=0; j<ZI.n_cols ; j+=1)
      {
        float ang = M_PI-((2.0 * M_PI * j )/(ZI.n_cols));

        if (ang < min_FOV-M_PI/2.0|| ang > max_FOV - M_PI/2.0) 
          continue;

        if(!(ZI(i,j)== 0 ))
        {  
          float pc_modulo = ZI(i,j);
          float pc_x = sqrt(pow(pc_modulo,2)- pow(ZzI(i,j),2)) * cos(ang);
          float pc_y = sqrt(pow(pc_modulo,2)- pow(ZzI(i,j),2)) * sin(ang);

          float ang_x_lidar = 0.6*M_PI/180.0;  

          Eigen::MatrixXf Lidar_matrix(3,3); //matrix  transformation between lidar and range image. It rotates the angles that it has of error with respect to the ground
          Eigen::MatrixXf result(3,1);
          Lidar_matrix <<   cos(ang_x_lidar) ,0                ,sin(ang_x_lidar),
                            0                ,1                ,0,
                            -sin(ang_x_lidar),0                ,cos(ang_x_lidar) ;

          result << pc_x,
                    pc_y,
                    ZzI(i,j);
          
          result = Lidar_matrix*result;  // rotacion en eje X para correccion

          point_cloud->points[num_pc].x = result(0);
          point_cloud->points[num_pc].y = result(1);
          point_cloud->points[num_pc].z = result(2);

          cloud->push_back(point_cloud->points[num_pc]); 

          num_pc++;
        }
      }
   }  

  //============================================================================================================

   PointCloud::Ptr P_out (new PointCloud);
   P_out = cloud;

  Eigen::MatrixXf RTlc(4,4); // translation matrix lidar-camera
  RTlc<<   Rlc(0), Rlc(3) , Rlc(6) ,Tlc(0)
          ,Rlc(1), Rlc(4) , Rlc(7) ,Tlc(1)
          ,Rlc(2), Rlc(5) , Rlc(8) ,Tlc(2)
          ,0       , 0        , 0  , 1    ;

  int size_inter_Lidar = (int) P_out->points.size();

  Eigen::MatrixXf Lidar_camera(3,size_inter_Lidar);
  Eigen::MatrixXf Lidar_cam(3,1);
  Eigen::MatrixXf pc_matrix(4,1);
  Eigen::MatrixXf pointCloud_matrix(4,size_inter_Lidar);

  unsigned int cols = in_image->width;
  unsigned int rows = in_image->height;

  uint px_data = 0; uint py_data = 0;

  pcl::PointXYZRGB point;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color (new pcl::PointCloud<pcl::PointXYZRGB>);

  for (int i = 0; i < size_inter_Lidar; i++)
  {
      pc_matrix(0,0) = -P_out->points[i].y;   
      pc_matrix(1,0) = -P_out->points[i].z;   
      pc_matrix(2,0) =  P_out->points[i].x;  
      pc_matrix(3,0) = 1.0;

      Lidar_cam = Mc * (RTlc * pc_matrix);

      px_data = (int)(Lidar_cam(0,0)/Lidar_cam(2,0));
      py_data = (int)(Lidar_cam(1,0)/Lidar_cam(2,0));
      
      if(px_data<0.0 || px_data>=cols || py_data<0.0 || py_data>=rows)
          continue;

      int color_dis_x = (int)(255*((P_out->points[i].x)/maxlen));
      int color_dis_z = (int)(255*((P_out->points[i].x)/10.0));
      if(color_dis_z>255)
          color_dis_z = 255;

      //point cloud con color
      cv::Vec3b & color = color_pcl->image.at<cv::Vec3b>(py_data,px_data);

      point.x = P_out->points[i].x;
      point.y = P_out->points[i].y;
      point.z = P_out->points[i].z;
      
      point.r = (int)color[2]; 
      point.g = (int)color[1]; 
      point.b = (int)color[0];
      
      pc_color->points.push_back(point);   
      
      cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1, CV_RGB(255-color_dis_x,(int)(color_dis_z),color_dis_x),cv::FILLED);
  }
  
  pc_color->is_dense = true;
  pc_color->width = (int) pc_color->points.size();
  pc_color->height = 1;
  pc_color->header.frame_id = "velodyne";

  pcOnimg_pub.publish(cv_ptr->toImageMsg());
  pc_pub.publish (pc_color);
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
  nh.getParam("/y_interpolation", interpol_value);
  nh.getParam("/ang_Y_resolution", angular_resolution_y);

  XmlRpc::XmlRpcValue param;

  nh.getParam("/matrix_file/tlc", param);
  Tlc <<  (double)param[0]
         ,(double)param[1]
         ,(double)param[2];

  nh.getParam("/matrix_file/rlc", param);

  Rlc <<  (double)param[0] ,(double)param[1] ,(double)param[2]
         ,(double)param[3] ,(double)param[4] ,(double)param[5]
         ,(double)param[6] ,(double)param[7] ,(double)param[8];

  nh.getParam("/matrix_file/camera_matrix", param);
  Mc  <<  (double)param[0] ,(double)param[1] ,(double)param[2] ,(double)param[3]
         ,(double)param[4] ,(double)param[5] ,(double)param[6] ,(double)param[7]
         ,(double)param[8] ,(double)param[9] ,(double)param[10],(double)param[11];

  message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic , 1);
  message_filters::Subscriber<Image> img_sub(nh, imgTopic, 1);

  typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);
  rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);

  pc_pub = nh.advertise<PointCloud> ("/points2", 1);  

  ros::spin();
}
