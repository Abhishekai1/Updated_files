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

// Function to calculate dynamic interpolation factor based on distance
float getDynamicInterpolationFactor(float distance) {
    if (distance <= 5.0f) {
        return 6.0f;  // 6 interpolated lines between layers for 0-5m
    } else if (distance <= 10.0f) {
        return 9.0f;  // 9 interpolated lines for 5-10m
    } else if (distance <= 15.0f) {
        return 12.0f; // 12 interpolated lines for 10-15m
    } else if (distance <= 20.0f) {
        return 15.0f; // 15 interpolated lines for 15-20m
    } else if (distance <= 25.0f) {
        return 18.0f; // 18 interpolated lines for 20-25m
    } else {
        return 20.0f; // 20 interpolated lines for >25m
    }
}

// Function to create dynamic interpolation grid
arma::mat createDynamicInterpolationGrid(const arma::mat& rangeData, const arma::mat& heightData, 
                                        int rows_img, int cols_img) {
    // Create distance map for each pixel
    arma::mat distanceMap = arma::zeros(rows_img, cols_img);
    
    // Calculate distance for each valid point
    for (int i = 0; i < rows_img; ++i) {
        for (int j = 0; j < cols_img; ++j) {
            if (rangeData(i, j) > 0) {
                distanceMap(i, j) = rangeData(i, j);
            }
        }
    }
    
    // Calculate maximum interpolation factor needed
    float maxInterpolFactor = 0.0f;
    for (int i = 0; i < rows_img; ++i) {
        for (int j = 0; j < cols_img; ++j) {
            if (distanceMap(i, j) > 0) {
                float interpol_factor = getDynamicInterpolationFactor(distanceMap(i, j));
                maxInterpolFactor = std::max(maxInterpolFactor, interpol_factor);
            }
        }
    }
    
    // Create interpolation pattern based on distance
    arma::vec X = arma::regspace(1, cols_img);
    arma::vec Y = arma::regspace(1, rows_img);
    arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
    arma::vec YI = arma::regspace(Y.min(), 1.0/maxInterpolFactor, Y.max());
    
    return arma::mat(YI.n_elem, XI.n_elem, arma::fill::zeros);
}

// Enhanced interpolation function with distance-based adaptive interpolation
void performDynamicInterpolation(arma::mat& ZI, arma::mat& ZzI, const arma::mat& Z, 
                                const arma::mat& Zz, int rows_img, int cols_img) {
    
    // Create distance-based interpolation grid
    arma::vec X = arma::regspace(1, cols_img);
    arma::vec Y = arma::regspace(1, rows_img);
    
    // Calculate adaptive interpolation factors
    arma::mat distanceMap = Z;
    float maxInterpolFactor = 0.0f;
    
    for (int i = 0; i < rows_img; ++i) {
        for (int j = 0; j < cols_img; ++j) {
            if (Z(i, j) > 0) {
                float interpol_factor = getDynamicInterpolationFactor(Z(i, j));
                maxInterpolFactor = std::max(maxInterpolFactor, interpol_factor);
            }
        }
    }
    
    // Use maximum interpolation factor for uniform grid
    arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
    arma::vec YI = arma::regspace(Y.min(), 1.0/maxInterpolFactor, Y.max());
    
    // Perform initial interpolation
    arma::interp2(X, Y, Z, XI, YI, ZI, "linear");
    arma::interp2(X, Y, Zz, XI, YI, ZzI, "linear");
    
    // Apply distance-based adaptive filtering
    arma::mat ZI_filtered = ZI;
    arma::mat ZzI_filtered = ZzI;
    
    for (uint i = 0; i < ZI.n_rows; ++i) {
        for (uint j = 0; j < ZI.n_cols; ++j) {
            if (ZI(i, j) > 0) {
                float distance = ZI(i, j);
                float local_interpol_factor = getDynamicInterpolationFactor(distance);
                
                // Apply distance-based smoothing
                if (local_interpol_factor > 6.0f) {
                    // For distant points, apply more aggressive smoothing
                    int window_size = std::min(5, (int)(local_interpol_factor / 4.0f));
                    double sum_range = 0.0;
                    double sum_height = 0.0;
                    int count = 0;
                    
                    for (int di = -window_size; di <= window_size; ++di) {
                        for (int dj = -window_size; dj <= window_size; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            
                            if (ni >= 0 && nj >= 0 && ni < (int)ZI.n_rows && nj < (int)ZI.n_cols) {
                                if (ZI(ni, nj) > 0) {
                                    double weight = 1.0 / (1.0 + std::sqrt(di*di + dj*dj));
                                    sum_range += ZI(ni, nj) * weight;
                                    sum_height += ZzI(ni, nj) * weight;
                                    count++;
                                }
                            }
                        }
                    }
                    
                    if (count > 0) {
                        ZI_filtered(i, j) = sum_range / count;
                        ZzI_filtered(i, j) = sum_height / count;
                    }
                }
            }
        }
    }
    
    ZI = ZI_filtered;
    ZzI = ZzI_filtered;
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
  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
  rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                       pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                       sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

  int cols_img = rangeImage->width;
  int rows_img = rangeImage->height;

  arma::mat Z;  // range image
  arma::mat Zz; // height image

  Z.zeros(rows_img,cols_img);         
  Zz.zeros(rows_img,cols_img);       

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

  ////////////////////////////////////////////// DYNAMIC INTERPOLATION
  //============================================================================================================
  
  arma::mat ZI;
  arma::mat ZzI;
  
  // Apply dynamic interpolation based on distance
  performDynamicInterpolation(ZI, ZzI, Z, Zz, rows_img, cols_img);

  //===========================================Enhanced processing=================================================
  
  // Handle zeros in interpolation with distance-aware approach
  arma::mat Zout = ZI;
  
  for (uint i = 0; i < ZI.n_rows; i++) {
      for (uint j = 0; j < ZI.n_cols; j++) {
          if (ZI(i, j) == 0) {
              float local_interpol_factor = 6.0f; // Default for unknown distance
              
              // Find nearest valid point to estimate distance
              for (int search_radius = 1; search_radius <= 10; search_radius++) {
                  bool found = false;
                  for (int di = -search_radius; di <= search_radius && !found; di++) {
                      for (int dj = -search_radius; dj <= search_radius && !found; dj++) {
                          int ni = i + di;
                          int nj = j + dj;
                          if (ni >= 0 && nj >= 0 && ni < (int)ZI.n_rows && nj < (int)ZI.n_cols) {
                              if (ZI(ni, nj) > 0) {
                                  local_interpol_factor = getDynamicInterpolationFactor(ZI(ni, nj));
                                  found = true;
                              }
                          }
                      }
                  }
                  if (found) break;
              }
              
              // Apply distance-based zero padding
              int padding_size = (int)local_interpol_factor;
              if (i + padding_size < ZI.n_rows) {
                  for (int k = 1; k <= padding_size; k++) {
                      Zout(i + k, j) = 0;
                  }
              }
              if (i > padding_size) {
                  for (int k = 1; k <= padding_size; k++) {
                      Zout(i - k, j) = 0;
                  }
              }
          }
      }
  }
  ZI = Zout;

  // Enhanced adaptive interpolation for missing data
  for (uint i = 1; i < ZI.n_rows - 1; ++i) {
      for (uint j = 1; j < ZI.n_cols - 1; ++j) {
          if (ZI(i, j) == 0) {
              double weighted_sum = 0.0;
              double weight_total = 0.0;
              
              // Estimate local interpolation factor
              float local_interpol_factor = 6.0f;
              for (int search_radius = 1; search_radius <= 5; search_radius++) {
                  for (int di = -search_radius; di <= search_radius; di++) {
                      for (int dj = -search_radius; dj <= search_radius; dj++) {
                          int ni = i + di;
                          int nj = j + dj;
                          if (ni >= 0 && nj >= 0 && ni < (int)ZI.n_rows && nj < (int)ZI.n_cols) {
                              if (ZI(ni, nj) > 0) {
                                  local_interpol_factor = getDynamicInterpolationFactor(ZI(ni, nj));
                                  goto found_reference;
                              }
                          }
                      }
                  }
              }
              found_reference:
              
              // Dynamic window size based on distance
              int window_size = std::max(3, std::min(9, (int)(local_interpol_factor / 2.0f)));
              
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

  // Enhanced edge preservation with distance-based thresholding
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

  double base_edge_threshold = 0.1 * arma::max(arma::max(grad_mag));

  for (uint i = 1; i < ZI.n_rows - 1; ++i) {
      for (uint j = 1; j < ZI.n_cols - 1; ++j) {
          if (ZI(i, j) > 0) {
              // Distance-based edge threshold
              float distance = ZI(i, j);
              float interpol_factor = getDynamicInterpolationFactor(distance);
              double edge_threshold = base_edge_threshold * (interpol_factor / 6.0f);
              
              if (grad_mag(i, j) > edge_threshold) {
                  double weight = std::max(0.0, 1.0 - grad_mag(i, j) / edge_threshold);
                  Zenhanced(i, j) = ZI(i, j) * weight + Zenhanced(i, j) * (1 - weight);
              }
          }
      }
  }

  ZI = Zenhanced;

  if (f_pc){    
      // Distance-based variance filtering
      for (uint i=0; i< ZI.n_rows; i+=1) {
          for (uint j=0; j<ZI.n_cols-5 ; j+=1) {
              if (ZI(i, j) > 0) {
                  float distance = ZI(i, j);
                  float local_interpol_factor = getDynamicInterpolationFactor(distance);
                  int filter_window = std::max(1, (int)(local_interpol_factor / 3.0f));
                  
                  double promedio = 0;
                  double varianza = 0;
                  int valid_count = 0;
                  
                  for (int k = 0; k < filter_window && (i + k) < ZI.n_rows; k++) {
                      if (ZI(i + k, j) > 0) {
                          promedio += ZI(i + k, j);
                          valid_count++;
                      }
                  }
                  
                  if (valid_count > 0) {
                      promedio = promedio / valid_count;
                      
                      for (int l = 0; l < filter_window && (i + l) < ZI.n_rows; l++) {
                          if (ZI(i + l, j) > 0) {
                              varianza += pow((ZI(i + l, j) - promedio), 2.0);
                          }
                      }
                      
                      // Distance-based variance threshold
                      double distance_var_threshold = max_var * (local_interpol_factor / 6.0f);
                      
                      if (varianza > distance_var_threshold) {
                          for (int m = 0; m < filter_window && (i + m) < ZI.n_rows; m++) {
                              Zout(i + m, j) = 0;
                          }
                      }
                  }
              }
          }
      }
      ZI = Zout;
  }

  ///////// Range image to point cloud with distance-aware processing
  PointCloud::Ptr point_cloud (new PointCloud);
  PointCloud::Ptr cloud (new PointCloud);
  point_cloud->width = ZI.n_cols; 
  point_cloud->height = ZI.n_rows;
  point_cloud->is_dense = false;
  point_cloud->points.resize (point_cloud->width * point_cloud->height);

  int num_pc = 0; 
  for (uint i=0; i< ZI.n_rows; i+=1) {       
      for (uint j=0; j<ZI.n_cols ; j+=1) {
          float ang = M_PI-((2.0 * M_PI * j )/(ZI.n_cols));

          if (ang < min_FOV-M_PI/2.0|| ang > max_FOV - M_PI/2.0) 
            continue;

          if(!(Zout(i,j)== 0 )) {  
            float pc_modulo = Zout(i,j);
            float pc_x = sqrt(pow(pc_modulo,2)- pow(ZzI(i,j),2)) * cos(ang);
            float pc_y = sqrt(pow(pc_modulo,2)- pow(ZzI(i,j),2)) * sin(ang);

            float ang_x_lidar = 0.6*M_PI/180.0;  

            Eigen::MatrixXf Lidar_matrix(3,3);
            Eigen::MatrixXf result(3,1);
            Lidar_matrix <<   cos(ang_x_lidar) ,0                ,sin(ang_x_lidar),
                              0                ,1                ,0,
                              -sin(ang_x_lidar),0                ,cos(ang_x_lidar) ;

            result << pc_x,
                      pc_y,
                      ZzI(i,j);
            
            result = Lidar_matrix*result;

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

   // Rest of the code remains the same...
   Eigen::MatrixXf RTlc(4,4);
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

   for (int i = 0; i < size_inter_Lidar; i++) {
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
