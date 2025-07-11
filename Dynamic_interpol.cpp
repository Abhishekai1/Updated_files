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

float interpol_value = 20.0; // This will be used as base interpolation value

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

// VLP-16 specific parameters
const int VLP16_LAYERS = 16;
const int BASE_INTERPOL = 7;  // Starting interpolation value for first layer
const int INTERPOL_INCREMENT = 2;  // Increment for each layer

// Function to get dynamic interpolation value for each layer
int getDynamicInterpolValue(int layer) {
    return BASE_INTERPOL + (layer * INTERPOL_INCREMENT);
}

// Function to get layer from row index
int getLayerFromRow(int row, int total_rows) {
    return (row * VLP16_LAYERS) / total_rows;
}

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
  ///

  ////// filter point cloud 
  if (msg_pointCloud == NULL) return;

  PointCloud::Ptr cloud_in (new PointCloud);
  //PointCloud::Ptr cloud_filter (new PointCloud);
  PointCloud::Ptr cloud_out (new PointCloud);

  //PointCloud::Ptr cloud_aux (new PointCloud);
 // pcl::PointXYZI point_aux;

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
       
       // Eigen::Vector3f tmp_point;
        //rangeImage->calculate3DPoint (float(i), float(j), r, tmp_point);
        if(std::isinf(r) || r<minlen || r>maxlen || std::isnan(zz)){
            continue;
        }             
        Z.at(j,i) = r;   
        Zz.at(j,i) = zz;
        //ZZei(j,i)=tmp_point[2];


        //point_aux.x = tmp_point[0];
        //point_aux.y = tmp_point[1];
        //point_aux.z = tmp_point[2];
      
       // cloud_aux->push_back(point_aux);



        //std::cout<<"i: "<<i<<" Z.getpoint: "<<zz<<" tmpPoint: "<<tmp_point<<std::endl;
       
      }

  ////////////////////////////////////////////// interpolation with dynamic values per layer
  //============================================================================================================
  
  arma::vec X = arma::regspace(1, Z.n_cols);  // X = horizontal spacing
  arma::vec Y = arma::regspace(1, Z.n_rows);  // Y = vertical spacing 

  // Calculate maximum interpolation value for determining final image size
  int max_interpol = getDynamicInterpolValue(VLP16_LAYERS - 1);
  
  arma::vec XI = arma:: regspace(X.min(), 1.0, X.max()); // magnify by approx 2
  arma::vec YI = arma::regspace(Y.min(), 1.0/max_interpol, Y.max()); // Use max interpolation for spacing

  arma::mat ZI_near;  
  arma::mat ZI;
  arma::mat ZzI;

  arma::interp2(X, Y, Z, XI, YI, ZI,"lineal");  
  arma::interp2(X, Y, Zz, XI, YI, ZzI,"lineal");  

  //===========================================fin filtrado por imagen=================================================
  /////////////////////////////

  // reconstruccion de imagen a nube 3D
  //============================================================================================================
  

  PointCloud::Ptr point_cloud (new PointCloud);
  PointCloud::Ptr cloud (new PointCloud);
  point_cloud->width = ZI.n_cols; 
  point_cloud->height = ZI.n_rows;
  point_cloud->is_dense = false;
  point_cloud->points.resize (point_cloud->width * point_cloud->height);

  arma::mat Zout = ZI;
  
  
  //////////////////filtrado de elementos interpolados con el fondo - Dynamic interpolation per layer
  // Handle zeros in interpolation with dynamic interpolation values
  for (uint i = 0; i < ZI.n_rows; i++) {
      int current_layer = getLayerFromRow(i, ZI.n_rows);
      int dynamic_interpol = getDynamicInterpolValue(current_layer);
      
      for (uint j = 0; j < ZI.n_cols; j++) {
          if (ZI(i, j) == 0) {
              if (i + dynamic_interpol < ZI.n_rows) {
                  for (int k = 1; k <= dynamic_interpol; k++) {
                      Zout(i + k, j) = 0;
                  }
              }
              if (i > dynamic_interpol) {
                  for (int k = 1; k <= dynamic_interpol; k++) {
                      Zout(i - k, j) = 0;
                  }
              }
          }
      }
  }
  ZI = Zout;

// Improved adaptive interpolation with layer-specific parameters
double base_density_threshold = 0.05;
int max_window_size = 11;
int min_window_size = 3;

for (uint i = 1; i < ZI.n_rows - 1; ++i) {
    int current_layer = getLayerFromRow(i, ZI.n_rows);
    int dynamic_interpol = getDynamicInterpolValue(current_layer);
    
    // Adjust density threshold based on layer interpolation value
    double density_threshold = base_density_threshold * (dynamic_interpol / (double)BASE_INTERPOL);
    
    for (uint j = 1; j < ZI.n_cols - 1; ++j) {
        if (ZI(i, j) == 0) {  // Missing data
            double weighted_sum = 0.0;
            double weight_total = 0.0;

            // Compute local density with adaptive window size
            int adaptive_search_window = std::min(dynamic_interpol / 2, 5);
            int valid_neighbors = 0;
            int total_neighbors = 0;
            
            for (int di = -adaptive_search_window; di <= adaptive_search_window; ++di) {
                for (int dj = -adaptive_search_window; dj <= adaptive_search_window; ++dj) {
                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols) {
                        total_neighbors++;
                        if (ZI(ni, nj) > 0) {
                            valid_neighbors++;
                        }
                    }
                }
            }

            double local_density = (double)valid_neighbors / total_neighbors;

            // Determine window size based on local density and layer characteristics
            int layer_adjusted_max_window = std::min(max_window_size, dynamic_interpol);
            int window_size = (local_density < density_threshold) ? layer_adjusted_max_window : min_window_size;

            // Enhanced interpolation using the determined window size
            for (int di = -window_size; di <= window_size; ++di) {
                for (int dj = -window_size; dj <= window_size; ++dj) {
                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                        double distance = std::sqrt(di * di + dj * dj);
                        
                        // Layer-aware weight calculation
                        double layer_weight_factor = 1.0 + (current_layer * 0.1); // Slightly increase weights for higher layers
                        double weight = layer_weight_factor / (distance + 1e-6);
                        
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

// Enhanced edge preservation with layer-specific processing
arma::mat Zenhanced = ZI;
arma::mat grad_x = arma::zeros(ZI.n_rows, ZI.n_cols);
arma::mat grad_y = arma::zeros(ZI.n_rows, ZI.n_cols);
arma::mat grad_mag = arma::zeros(ZI.n_rows, ZI.n_cols);

// Calculate gradients with layer-aware smoothing
for (uint i = 1; i < ZI.n_rows - 1; ++i) {
    int current_layer = getLayerFromRow(i, ZI.n_rows);
    int dynamic_interpol = getDynamicInterpolValue(current_layer);
    
    for (uint j = 1; j < ZI.n_cols - 1; ++j) {
        if (ZI(i, j) > 0) {
            // Use layer-specific gradient calculation
            double layer_factor = (double)dynamic_interpol / BASE_INTERPOL;
            int grad_window = std::max(1, (int)(layer_factor * 0.5));
            
            if (j >= grad_window && j < ZI.n_cols - grad_window) {
                grad_x(i, j) = (ZI(i, j + grad_window) - ZI(i, j - grad_window)) / (2.0 * grad_window);
            }
            if (i >= grad_window && i < ZI.n_rows - grad_window) {
                grad_y(i, j) = (ZI(i + grad_window, j) - ZI(i - grad_window, j)) / (2.0 * grad_window);
            }
            
            grad_mag(i, j) = std::sqrt(grad_x(i, j) * grad_x(i, j) + grad_y(i, j) * grad_y(i, j));
        }
    }
}

// Layer-adaptive edge threshold
for (uint i = 1; i < ZI.n_rows - 1; ++i) {
    int current_layer = getLayerFromRow(i, ZI.n_rows);
    int dynamic_interpol = getDynamicInterpolValue(current_layer);
    
    // Calculate layer-specific edge threshold
    double layer_threshold_factor = 0.08 + (current_layer * 0.02 / VLP16_LAYERS);
    double local_max_grad = 0.0;
    
    // Find local maximum gradient in layer neighborhood
    int layer_start = std::max(0, (int)(i - dynamic_interpol/2));
    int layer_end = std::min((int)ZI.n_rows - 1, (int)(i + dynamic_interpol/2));
    
    for (int li = layer_start; li <= layer_end; ++li) {
        for (uint lj = 0; lj < ZI.n_cols; ++lj) {
            if (grad_mag(li, lj) > local_max_grad) {
                local_max_grad = grad_mag(li, lj);
            }
        }
    }
    
    double edge_threshold = layer_threshold_factor * local_max_grad;
    
    for (uint j = 1; j < ZI.n_cols - 1; ++j) {
        if (grad_mag(i, j) > edge_threshold) {
            // Enhanced edge preservation with layer-specific parameters
            double preservation_strength = 0.7 + (current_layer * 0.2 / VLP16_LAYERS);
            double weight = std::max(0.0, 1.0 - (grad_mag(i, j) / edge_threshold) * preservation_strength);
            Zenhanced(i, j) = ZI(i, j) * weight + Zenhanced(i, j) * (1 - weight);
        }
    }
}

// Replace ZI with the enhanced version
ZI = Zenhanced;

  if (f_pc){    
    //////////////////filtrado de elementos interpolados con el fondo - Dynamic layer processing
    
    /// filtrado por varianza with dynamic interpolation values per layer
    for (uint i = 0; i < ZI.n_rows; i++) {
        int current_layer = getLayerFromRow(i, ZI.n_rows);
        int dynamic_interpol = getDynamicInterpolValue(current_layer);
        
        // Skip if we don't have enough rows for this layer's interpolation
        if (i + dynamic_interpol >= ZI.n_rows) continue;
        
        for (uint j = 0; j < ZI.n_cols - 5; j++) {
            double promedio = 0;
            double varianza = 0;
            
            for (int k = 0; k < dynamic_interpol; k++) {
                if (i + k < ZI.n_rows) {
                    promedio += ZI(i + k, j);
                }
            }
            
            promedio = promedio / dynamic_interpol;
            
            for (int l = 0; l < dynamic_interpol; l++) {
                if (i + l < ZI.n_rows) {
                    varianza += pow((ZI(i + l, j) - promedio), 2.0);
                }
            }
            
            // Layer-specific variance threshold
            double layer_max_var = max_var * (1.0 + current_layer * 0.1 / VLP16_LAYERS);
            
            if (varianza > layer_max_var) {
                for (int m = 0; m < dynamic_interpol; m++) {
                    if (i + m < ZI.n_rows) {
                        Zout(i + m, j) = 0;
                    }
                }
            }
        }
        
        // Skip ahead by the interpolation value for efficiency
        i += dynamic_interpol - 1;
    }
    ZI = Zout;
  }

  ///////// imagen de rango a nube de puntos with layer-aware processing
  int num_pc = 0; 
  for (uint i = 0; i < ZI.n_rows; i++) {
      int current_layer = getLayerFromRow(i, ZI.n_rows);
      int dynamic_interpol = getDynamicInterpolValue(current_layer);
      
      // Skip if we don't have enough rows
      if (i + dynamic_interpol >= ZI.n_rows) continue;
      
      for (uint j = 0; j < ZI.n_cols; j++) {
          float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));

          if (ang < min_FOV - M_PI/2.0 || ang > max_FOV - M_PI/2.0) 
              continue;

          if (!(Zout(i, j) == 0)) {
              float pc_modulo = Zout(i, j);
              float pc_x = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * cos(ang);
              float pc_y = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * sin(ang);

              float ang_x_lidar = 0.6 * M_PI / 180.0;

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

              cloud->push_back(point_cloud->points[num_pc]);
              num_pc++;
          }
      }
  }

  //============================================================================================================

   PointCloud::Ptr P_out (new PointCloud);
 
   //filremove noise of point cloud
  /*pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50.0);
  sor.setStddevMulThresh (1.0);
  sor.filter (*P_out);*/

  // dowsmapling
  /*pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.1f, 0.1f, 0.1f);
  sor.filter (*P_out);*/


  P_out = cloud;


  Eigen::MatrixXf RTlc(4,4); // translation matrix lidar-camera
  RTlc<<   Rlc(0), Rlc(3) , Rlc(6) ,Tlc(0)
          ,Rlc(1), Rlc(4) , Rlc(7) ,Tlc(1)
          ,Rlc(2), Rlc(5) , Rlc(8) ,Tlc(2)
          ,0       , 0        , 0  , 1    ;

  //std::cout<<RTlc<<std::endl;

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

   //P_out = cloud_out;

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
