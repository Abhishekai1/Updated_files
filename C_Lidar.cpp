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

// Enhanced interpolation function for far regions
arma::mat enhancedDistanceBasedInterpolation(const arma::mat& Z, const arma::mat& Zz, float max_distance) {
    arma::mat result = Z;
    
    // Distance-adaptive interpolation parameters
    for (uint i = 1; i < Z.n_rows - 1; ++i) {
        for (uint j = 1; j < Z.n_cols - 1; ++j) {
            if (Z(i, j) == 0) {  // Missing data point
                
                // Calculate average distance in neighborhood to determine if we're in far region
                double avg_distance = 0.0;
                int valid_neighbors = 0;
                
                for (int di = -2; di <= 2; ++di) {
                    for (int dj = -2; dj <= 2; ++dj) {
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni >= 0 && nj >= 0 && ni < Z.n_rows && nj < Z.n_cols && Z(ni, nj) > 0) {
                            avg_distance += Z(ni, nj);
                            valid_neighbors++;
                        }
                    }
                }
                
                if (valid_neighbors > 0) {
                    avg_distance /= valid_neighbors;
                    
                    // Adaptive window size based on distance - larger windows for far regions
                    int base_window = 3;
                    int adaptive_window = base_window;
                    
                    if (avg_distance > max_distance * 0.7) {  // Far region (>70% of max distance)
                        adaptive_window = std::min(15, base_window + (int)(6 * avg_distance / max_distance));
                    } else if (avg_distance > max_distance * 0.4) {  // Medium range
                        adaptive_window = base_window + 2;
                    }
                    
                    // Enhanced weighted interpolation
                    double weighted_sum = 0.0;
                    double weight_total = 0.0;
                    
                    for (int di = -adaptive_window; di <= adaptive_window; ++di) {
                        for (int dj = -adaptive_window; dj <= adaptive_window; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            
                            if (ni >= 0 && nj >= 0 && ni < Z.n_rows && nj < Z.n_cols && Z(ni, nj) > 0) {
                                double distance = std::sqrt(di * di + dj * dj);
                                
                                // Enhanced weighting function for far regions
                                double weight;
                                if (avg_distance > max_distance * 0.6) {
                                    // Smoother fall-off for far regions to create better continuity
                                    weight = std::exp(-distance * distance / (2.0 * adaptive_window * adaptive_window));
                                } else {
                                    // Standard inverse distance weighting for near regions
                                    weight = 1.0 / (distance + 1e-6);
                                }
                                
                                weighted_sum += Z(ni, nj) * weight;
                                weight_total += weight;
                            }
                        }
                    }
                    
                    if (weight_total > 0) {
                        result(i, j) = weighted_sum / weight_total;
                    }
                }
            }
        }
    }
    
    return result;
}

// Multi-scale interpolation for better line continuity in far regions
arma::mat multiScaleLineInterpolation(const arma::mat& Z, float max_distance) {
    arma::mat result = Z;
    
    // Apply multiple passes with different scales for better line continuity
    for (int scale = 1; scale <= 3; scale++) {
        arma::mat temp_result = result;
        
        for (uint i = scale; i < Z.n_rows - scale; i += 1) {
            for (uint j = scale; j < Z.n_cols - scale; j += 1) {
                
                if (result(i, j) == 0) {
                    // Check for line patterns in multiple directions
                    std::vector<std::pair<double, double>> line_candidates;
                    
                    // Horizontal line detection and interpolation
                    if (result(i, j - scale) > 0 && result(i, j + scale) > 0) {
                        double avg_dist = (result(i, j - scale) + result(i, j + scale)) / 2.0;
                        if (avg_dist > max_distance * 0.5) {  // Prioritize far region lines
                            line_candidates.push_back({avg_dist, 1.5}); // Higher weight for distance-based priority
                        } else {
                            line_candidates.push_back({avg_dist, 1.0});
                        }
                    }
                    
                    // Vertical line detection and interpolation
                    if (result(i - scale, j) > 0 && result(i + scale, j) > 0) {
                        double avg_dist = (result(i - scale, j) + result(i + scale, j)) / 2.0;
                        if (avg_dist > max_distance * 0.5) {
                            line_candidates.push_back({avg_dist, 1.5});
                        } else {
                            line_candidates.push_back({avg_dist, 1.0});
                        }
                    }
                    
                    // Diagonal line detection (important for object edges in far regions)
                    if (result(i - scale, j - scale) > 0 && result(i + scale, j + scale) > 0) {
                        double avg_dist = (result(i - scale, j - scale) + result(i + scale, j + scale)) / 2.0;
                        if (avg_dist > max_distance * 0.5) {
                            line_candidates.push_back({avg_dist, 1.2});
                        } else {
                            line_candidates.push_back({avg_dist, 0.8});
                        }
                    }
                    
                    if (result(i - scale, j + scale) > 0 && result(i + scale, j - scale) > 0) {
                        double avg_dist = (result(i - scale, j + scale) + result(i + scale, j - scale)) / 2.0;
                        if (avg_dist > max_distance * 0.5) {
                            line_candidates.push_back({avg_dist, 1.2});
                        } else {
                            line_candidates.push_back({avg_dist, 0.8});
                        }
                    }
                    
                    // Select best interpolation candidate
                    if (!line_candidates.empty()) {
                        double weighted_sum = 0.0;
                        double weight_sum = 0.0;
                        
                        for (const auto& candidate : line_candidates) {
                            weighted_sum += candidate.first * candidate.second;
                            weight_sum += candidate.second;
                        }
                        
                        if (weight_sum > 0) {
                            temp_result(i, j) = weighted_sum / weight_sum;
                        }
                    }
                }
            }
        }
        result = temp_result;
    }
    
    return result;
}

// Adaptive density enhancement for far regions
PointCloud::Ptr enhanceFarRegionDensity(PointCloud::Ptr input_cloud, float distance_threshold) {
    PointCloud::Ptr enhanced_cloud(new PointCloud);
    
    for (size_t i = 0; i < input_cloud->points.size(); i++) {
        pcl::PointXYZI original_point = input_cloud->points[i];
        enhanced_cloud->push_back(original_point);
        
        // Calculate distance from sensor
        double distance = sqrt(original_point.x * original_point.x + 
                              original_point.y * original_point.y + 
                              original_point.z * original_point.z);
        
        // Add additional interpolated points for far regions
        if (distance > distance_threshold) {
            // Look for nearby points to create interpolated points
            for (size_t j = i + 1; j < std::min(i + 10, input_cloud->points.size()); j++) {
                pcl::PointXYZI neighbor = input_cloud->points[j];
                
                double neighbor_distance = sqrt(neighbor.x * neighbor.x + 
                                               neighbor.y * neighbor.y + 
                                               neighbor.z * neighbor.z);
                
                // If neighbor is also in far region and close in 3D space
                if (neighbor_distance > distance_threshold) {
                    double spatial_distance = sqrt(pow(original_point.x - neighbor.x, 2) +
                                                  pow(original_point.y - neighbor.y, 2) +
                                                  pow(original_point.z - neighbor.z, 2));
                    
                    // Create intermediate points for better density
                    if (spatial_distance > 0.5 && spatial_distance < 3.0) {
                        pcl::PointXYZI intermediate;
                        intermediate.x = (original_point.x + neighbor.x) * 0.5;
                        intermediate.y = (original_point.y + neighbor.y) * 0.5;
                        intermediate.z = (original_point.z + neighbor.z) * 0.5;
                        intermediate.intensity = (original_point.intensity + neighbor.intensity) * 0.5;
                        
                        enhanced_cloud->push_back(intermediate);
                    }
                }
            }
        }
    }
    
    return enhanced_cloud;
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

  ////////////////////////////////////////////// Enhanced interpolation
  //============================================================================================================
  
  arma::vec X = arma::regspace(1, Z.n_cols);  // X = horizontal spacing
  arma::vec Y = arma::regspace(1, Z.n_rows);  // Y = vertical spacing 

  

  arma::vec XI = arma:: regspace(X.min(), 1.0, X.max()); // magnify by approx 2
  arma::vec YI = arma::regspace(Y.min(), 1.0/interpol_value, Y.max()); // 


  arma::mat ZI_near;  
  arma::mat ZI;
  arma::mat ZzI;

  arma::interp2(X, Y, Z, XI, YI, ZI,"lineal");  
  arma::interp2(X, Y, Zz, XI, YI, ZzI,"lineal");  

  // Apply enhanced distance-based interpolation
  ZI = enhancedDistanceBasedInterpolation(ZI, ZzI, maxlen);
  
  // Apply multi-scale line interpolation for better connectivity in far regions
  ZI = multiScaleLineInterpolation(ZI, maxlen);

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
  
  
  //////////////////filtrado de elementos interpolados con el fondo
  // Handle zeros in interpolation (improved with distance awareness)
for (uint i = 0; i < ZI.n_rows; i++) {
    for (uint j = 0; j < ZI.n_cols; j++) {
        if (ZI(i, j) == 0) {
            // Calculate average distance in neighborhood
            double avg_nearby_distance = 0.0;
            int nearby_count = 0;
            
            for (int di = -2; di <= 2; di++) {
                for (int dj = -2; dj <= 2; dj++) {
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                        avg_nearby_distance += ZI(ni, nj);
                        nearby_count++;
                    }
                }
            }
            
            if (nearby_count > 0) {
                avg_nearby_distance /= nearby_count;
                
                // Adaptive masking based on distance - be more aggressive for far regions
                int mask_range = interpol_value;
                if (avg_nearby_distance > maxlen * 0.6) {
                    mask_range = interpol_value / 2;  // Less aggressive masking for far regions
                }
                
                if (i + mask_range < ZI.n_rows) {
                    for (int k = 1; k <= mask_range; k++) {
                        Zout(i + k, j) = 0;
                    }
                }
                if (i > mask_range) {
                    for (int k = 1; k <= mask_range; k++) {
                        Zout(i - k, j) = 0;
                    }
                }
            }
        }
    }
}
ZI = Zout;

// Enhanced adaptive interpolation with better density control
double density_threshold = 0.05;  
int max_window_size = 12;  // Increased for better far region coverage
int min_window_size = 3;  

for (uint i = 1; i < ZI.n_rows - 1; ++i) {
    for (uint j = 1; j < ZI.n_cols - 1; ++j) {
        if (ZI(i, j) == 0) {  // Missing data
            double weighted_sum = 0.0;
            double weight_total = 0.0;

            // Compute local density and average distance
            int valid_neighbors = 0;
            double avg_distance = 0.0;
            
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                        valid_neighbors++;
                        avg_distance += ZI(ni, nj);
                    }
                }
            }
            
            if (valid_neighbors > 0) {
                avg_distance /= valid_neighbors;
                
                // Enhanced window size determination
                int window_size = min_window_size;
                if (avg_distance > maxlen * 0.7) {  // Far region
                    window_size = max_window_size;
                } else if (avg_distance > maxlen * 0.4) {  // Medium range
                    window_size = (max_window_size + min_window_size) / 2;
                } else if (valid_neighbors < density_threshold * 25) {  // Low density area
                    window_size = max_window_size;
                }

                // Enhanced weighted interpolation
                for (int di = -window_size; di <= window_size; ++di) {
                    for (int dj = -window_size; dj <= window_size; ++dj) {
                        int ni = i + di;
                        int nj = j + dj;

                        if (ni >= 0 && nj >= 0 && ni < ZI.n_rows && nj < ZI.n_cols && ZI(ni, nj) > 0) {
                            double distance = std::sqrt(di * di + dj * dj);
                            
                            // Enhanced weighting with distance-based adaptation
                            double weight;
                            if (avg_distance > maxlen * 0.6) {
                                // Gaussian-like weighting for far regions (smoother)
                                weight = std::exp(-distance * distance / (2.0 * window_size * window_size / 4.0));
                            } else {
                                // Standard inverse distance weighting
                                weight = 1.0 / (distance + 1e-6);
                            }
                            
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
}



// Enhanced edge preservation with distance awareness
  arma::mat Zenhanced = ZI;
  arma::mat grad_x = arma::zeros(ZI.n_rows, ZI.n_cols);
  arma::mat grad_y = arma::zeros(ZI.n_rows, ZI.n_cols);
  arma::mat grad_mag = arma::zeros(ZI.n_rows, ZI.n_cols);

// Calculate gradients with enhanced far region handling
for (uint i = 1; i < ZI.n_rows - 1; ++i) {
    for (uint j = 1; j < ZI.n_cols - 1; ++j) {
        if (ZI(i, j) > 0) {
            grad_x(i, j) = (ZI(i, j + 1) - ZI(i, j - 1)) * 0.5;
            grad_y(i, j) = (ZI(i + 1, j) - ZI(i - 1, j)) * 0.5;
            grad_mag(i, j) = std::sqrt(grad_x(i, j) * grad_x(i, j) + grad_y(i, j) * grad_y(i, j));
        }
    }
}

// Adaptive edge threshold based on distance
for (uint i = 1; i < ZI.n_rows - 1; ++i) {
    for (uint j = 1; j < ZI.n_cols - 1; ++j) {
        if (ZI(i, j) > 0) {
            double distance = ZI(i, j);
            double edge_threshold;
            
            if (distance > maxlen * 0.7) {
                // More relaxed edge preservation for far regions to maintain connectivity
                edge_threshold = 0.05 * arma::max(arma::max(grad_mag));
            } else {
                // Standard edge preservation for near regions
                edge_threshold = 0.1 * arma::max(arma::max(grad_mag));
            }
            
            if (grad_mag(i, j) > edge_threshold) {
                double weight = std::max(0.0, 1.0 - (grad_mag(i, j) - edge_threshold) / edge_threshold);
                Zenhanced(i, j) = ZI(i, j) * (1.0 - weight) + Zenhanced(i, j) * weight;
            }
        }
    }
}

ZI = Zenhanced;



  if (f_pc){    
    //////////////////filtrado de elementos interpolados con el fondo
    
    /// Enhanced variance filtering with distance adaptation
  for (uint i=0; i< ((ZI.n_rows-1)/interpol_value); i+=1)       
      for (uint j=0; j<ZI.n_cols-5 ; j+=1)
      {
        double promedio = 0;
        double varianza = 0;
        double avg_distance = 0;
        int valid_points = 0;
        
        for (uint k=0; k<interpol_value ; k+=1) {
            if (ZI((i*interpol_value)+k,j) > 0) {
                promedio += ZI((i*interpol_value)+k,j);
                avg_distance += ZI((i*interpol_value)+k,j);
                valid_points++;
            }
        }

        if (valid_points > 0) {
            promedio = promedio / valid_points;    
            avg_distance = avg_distance / valid_points;

            for (uint l = 0; l < interpol_value; l++) {
                if (ZI((i*interpol_value)+l,j) > 0) {
                    varianza = varianza + pow((ZI((i*interpol_value)+l,j) - promedio), 2.0);  
                }
            }
            
            // Adaptive variance threshold - more lenient for far regions
            double adaptive_max_var = max_var;
            if (avg_distance > maxlen * 0.6) {
                adaptive_max_var = max_var * 2.0;  // Allow more variance in far regions
            }

            if(varianza > adaptive_max_var) {
                for (uint m = 0; m < interpol_value; m++) {
                    Zout((i*interpol_value)+m,j) = 0;                 
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

  // Apply enhanced density enhancement for far regions
  P_out = enhanceFarRegionDensity(cloud, maxlen * 0.6);


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
      
      // Enhanced visualization with distance-based circle sizes for better far region visibility
      double distance = sqrt(P_out->points[i].x * P_out->points[i].x + 
                            P_out->points[i].y * P_out->points[i].y + 
                            P_out->points[i].z * P_out->points[i].z);
      
      int circle_radius = 1;
      if (distance > maxlen * 0.7) {
          circle_radius = 2;  // Larger circles for far region points
      } else if (distance > maxlen * 0.4) {
          circle_radius = 1;
      }
      
      cv::circle(cv_ptr->image, cv::Point(px_data, py_data), circle_radius, 
                CV_RGB(255-color_dis_x,(int)(color_dis_z),color_dis_x), cv::FILLED);
      
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
