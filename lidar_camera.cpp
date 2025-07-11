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

float interpol_value = 20.0; // Base interpolation value

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

// Function to calculate dynamic interpolation value for VLP-16
int getDynamicInterpolationValue(int layer_index, int total_layers) {
    // For VLP-16: first 5 layers use 7 interpolated lines
    // After that, increase by 2 for each subsequent layer: 9, 11, 13, etc.
    if (layer_index < 5) {
        return 7;
    } else {
        return 7 + 2 * (layer_index - 4);
    }
}

// Function to get layer index based on row position
int getLayerIndex(int row, int total_rows) {
    // Assuming 16 layers for VLP-16, distribute rows evenly
    return (row * 16) / total_rows;
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

  ////////////////////////////////////////////// Dynamic interpolation for VLP-16
  //============================================================================================================
  
  arma::vec X = arma::regspace(1, Z.n_cols);  // X = horizontal spacing
  arma::vec Y = arma::regspace(1, Z.n_rows);  // Y = vertical spacing 

  // Calculate maximum interpolation factor needed
  int max_layer = getLayerIndex(Z.n_rows - 1, Z.n_rows);
  int max_interpol = getDynamicInterpolationValue(max_layer, 16);
  
  arma::vec XI = arma:: regspace(X.min(), 1.0, X.max()); // magnify by approx 2
  arma::vec YI = arma::regspace(Y.min(), 1.0/max_interpol, Y.max()); // Use maximum interpolation

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
  
  
  // Enhanced dynamic interpolation processing with layer-specific handling
  for (uint i = 0; i < ZI.n_rows; i++) {
      int current_layer = getLayerIndex(i, ZI.n_rows);
      int current_interpol = getDynamicInterpolationValue(current_layer, 16);
      
      for (uint j = 0; j < ZI.n_cols; j++) {
          if (ZI(i, j) == 0) {
              if (i + current_interpol < ZI.n_rows) {
                  for (int k = 1; k <= current_interpol; k++) {
                      Zout(i + k, j) = 0;
                  }
              }
              if (i > current_interpol) {
                  for (int k = 1; k <= current_interpol; k++) {
                      Zout(i - k, j) = 0;
                  }
              }
          }
      }
  }
  ZI = Zout;

  // Improved adaptive interpolation with layer-specific parameters
  for (uint i = 1; i < ZI.n_rows - 1; ++i) {
      int current_layer = getLayerIndex(i, ZI.n_rows);
      int current_interpol = getDynamicInterpolationValue(current_layer, 16);
      
      // Adaptive parameters based on layer
      double density_threshold = (current_layer < 5) ? 0.08 : 0.05;  // Higher threshold for first 5 layers
      int max_window_size = std::min(15, current_interpol + 2);  // Scale window with interpolation
      int min_window_size = std::max(3, current_interpol / 3);   // Minimum window scales too
      
      for (uint j = 1; j < ZI.n_cols - 1; ++j) {
          if (ZI(i, j) == 0) {  // Missing data
              double weighted_sum = 0.0;
              double weight_total = 0.0;

              // Compute local density with adaptive window
              int valid_neighbors = 0;
              int density_window = std::min(5, current_interpol / 2);
              
              for (int di = -density_window; di <= density_window; ++di) {
                  for (int dj = -density_window; dj <= density_window; ++dj) {
                      int ni = i + di;
                      int nj = j + dj;

                      if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                          valid_neighbors++;
                      }
                  }
              }

              // Determine window size based on local density and layer
              int total_neighbors = (2 * density_window + 1) * (2 * density_window + 1);
              double local_density = (double)valid_neighbors / total_neighbors;
              int window_size = (local_density < density_threshold) ? max_window_size : min_window_size;

              // Layer-specific weight adjustment
              double layer_weight_factor = (current_layer < 5) ? 1.2 : 1.0;

              // Interpolate using the determined window size
              for (int di = -window_size; di <= window_size; ++di) {
                  for (int dj = -window_size; dj <= window_size; ++dj) {
                      int ni = i + di;
                      int nj = j + dj;

                      if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                          double distance = std::sqrt(di * di + dj * dj);
                          double weight = (1.0 / (distance + 1e-6)) * layer_weight_factor;
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

  // Enhanced edge preservation with layer-specific parameters
  arma::mat Zenhanced = ZI;
  arma::mat grad_x = arma::zeros(ZI.n_rows, ZI.n_cols);
  arma::mat grad_y = arma::zeros(ZI.n_rows, ZI.n_cols);
  arma::mat grad_mag = arma::zeros(ZI.n_rows, ZI.n_cols);

  // Calculate gradients with layer-specific smoothing
  for (uint i = 1; i < ZI.n_rows - 1; ++i) {
      int current_layer = getLayerIndex(i, ZI.n_rows);
      double smoothing_factor = (current_layer < 5) ? 0.5 : 0.3; // Less smoothing for first 5 layers
      
      for (uint j = 1; j < ZI.n_cols - 1; ++j) {
          if (ZI(i, j) > 0) {  // Only consider valid points
              grad_x(i, j) = (ZI(i, j + 1) - ZI(i, j - 1)) * smoothing_factor;
              grad_y(i, j) = (ZI(i + 1, j) - ZI(i - 1, j)) * smoothing_factor;
              grad_mag(i, j) = std::sqrt(grad_x(i, j) * grad_x(i, j) + grad_y(i, j) * grad_y(i, j));
          }
      }
  }

  // Layer-specific edge thresholds
  for (uint i = 1; i < ZI.n_rows - 1; ++i) {
      int current_layer = getLayerIndex(i, ZI.n_rows);
      double edge_threshold_factor = (current_layer < 5) ? 0.12 : 0.08;  // Higher threshold for first 5 layers
      double edge_threshold = edge_threshold_factor * arma::max(arma::max(grad_mag));
      
      for (uint j = 1; j < ZI.n_cols - 1; ++j) {
          if (grad_mag(i, j) > edge_threshold) {
              double weight = std::max(0.0, 1.0 - grad_mag(i, j) / edge_threshold);
              // Apply stronger edge preservation for higher layers
              double preservation_strength = (current_layer < 5) ? 0.8 : 0.9;
              Zenhanced(i, j) = ZI(i, j) * weight + Zenhanced(i, j) * (1 - weight) * preservation_strength;
          }
      }
  }

  // Replace ZI with the enhanced version
  ZI = Zenhanced;

  if (f_pc){    
    // Dynamic variance filtering based on layer-specific interpolation
    for (uint i = 0; i < ZI.n_rows; i++) {
        int current_layer = getLayerIndex(i, ZI.n_rows);
        int current_interpol = getDynamicInterpolationValue(current_layer, 16);
        
        // Process in chunks based on current interpolation value
        if (i + current_interpol < ZI.n_rows) {
            for (uint j = 0; j < ZI.n_cols - 5; j++) {
                double promedio = 0;
                double varianza = 0;
                
                for (int k = 0; k < current_interpol; k++) {
                    promedio += ZI(i + k, j);
                }
                promedio = promedio / current_interpol;
                
                for (int l = 0; l < current_interpol; l++) {
                    varianza += pow((ZI(i + l, j) - promedio), 2.0);
                }
                
                // Layer-specific variance threshold
                double layer_max_var = (current_layer < 5) ? max_var * 1.2 : max_var;
                
                if (varianza > layer_max_var) {
                    for (int m = 0; m < current_interpol; m++) {
                        Zout(i + m, j) = 0;
                    }
                }
            }
        }
    }
    ZI = Zout;
  }

  ///////// imagen de rango a nube de puntos  
  int num_pc = 0; 
  for (uint i=0; i< ZI.n_rows - interpol_value; i+=1)
   {       
      for (uint j=0; j<ZI.n_cols ; j+=1)
      {

        float ang = M_PI-((2.0 * M_PI * j )/(ZI.n_cols));

        if (ang < min_FOV-M_PI/2.0|| ang > max_FOV - M_PI/2.0) 
          continue;

        if(!(Zout(i,j)== 0 ))
        {  
          float pc_modulo = Zout(i,j);
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
