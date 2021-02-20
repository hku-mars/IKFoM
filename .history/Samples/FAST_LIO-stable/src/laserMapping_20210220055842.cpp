// This is a modification of the algorithm described in the following paper:
//W.  Xu  and  F.  Zhang. Fast-lio:  A  fast,  robust  lidar-inertial  odome-try  package  by  tightly-coupled  iterated  kalman  filter. 
//arXiv  preprintarXiv:2010.08196, 2020

/*
// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
*/
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <Exp_mat.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <opencv/cv.h>
#include <common_lib.h>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/core/eigen.hpp>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <fast_lio/States.h>
#include <geometry_msgs/Vector3.h>


#ifndef DEPLOY
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

#define Modified

#define INIT_TIME           (1.0)
#define LASER_POINT_COV     (0.0015)
#define NUM_MATCH_POINTS    (5)

std::string root_dir = ROOT_DIR;

int iterCount = 0;
int NUM_MAX_ITERATIONS  = 0;
int laserCloudCenWidth  = 20;
int laserCloudCenHeight = 10;
int laserCloudCenDepth  = 20;
int laserCloudValidInd[250];
int laserCloudSurroundInd[250];
int laserCloudValidNum    = 0;
int laserCloudSurroundNum = 0;
int laserCloudSelNum      = 0;

const int laserCloudWidth  = 42;
const int laserCloudHeight = 22;
const int laserCloudDepth  = 42;
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;

/// IMU relative variables
std::mutex mtx_buffer;
std::condition_variable sig_buffer;
bool lidar_pushed = false;
bool b_exit = false;
bool b_reset = false;
bool b_first = true;

/// Buffers for measurements
double cube_len = 0.0;
double lidar_end_time = 0.0;
double last_timestamp_lidar = -1;
double last_timestamp_imu   = -1;
double HALF_FOV_COS = 0.0;
double res_mean_last = 0.05;

std::deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

//surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_updated (new PointCloudXYZI(*feats_down));
std::vector<bool> point_selected_surf;
std::vector<std::vector<int>> pointSearchInd_surf;
std::vector<double> res_last; // initial

std::deque< fast_lio::States > rot_kp_imu_buff;

//all points
PointCloudXYZI::Ptr laserCloudFullRes2(new PointCloudXYZI());
PointCloudXYZI::Ptr featsArray[laserCloudNum];
// bool                _last_inFOV[laserCloudNum];
pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudFullResColor(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

//estimator inputs and output;
MeasureGroup Measures;
//StatesGroup  state;
ekf state_cal;

//project lidar frame to world
void pointBodyToWorld(PointType const * const pi, PointType * const po, state &s)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I)+ s.pos);
    //Eigen::Vector3d p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// Modified measurement model and its differentions
#ifdef Modified
Eigen::Matrix<double, Eigen::Dynamic, 1> get_h_modified(state &s, esekfom::dyn_cal<double> &dyn_share)
{
                    PointCloudXYZI::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
                    PointCloudXYZI::Ptr coeffSel(new pcl::PointCloud<PointType>());
                    int laserCloudSurf_down_size = feats_down->points.size();
                    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
                    PointCloudXYZI::Ptr coeffSel_tmpt(new pcl::PointCloud<PointType>(*feats_down));              
                    kdtreeSurfFromMap->setInputCloud(featsFromMap);
                    /** closest surface search and seclection **/
                    omp_set_num_threads(4);
                    #pragma omp parallel for
                    for (int i = 0; i < laserCloudSurf_down_size; i++)
                    {
                        PointType &pointOri_tmpt = feats_down->points[i];
                        PointType &pointSel_tmpt = feats_down_updated->points[i];
                        pointBodyToWorld(&pointOri_tmpt, &pointSel_tmpt, s);

                        std::vector<float> pointSearchSqDis_surf;
                        auto &points_near = pointSearchInd_surf[i];
                        if (dyn_share.converge)
                        {
                            point_selected_surf[i] = true;
                            kdtreeSurfFromMap->nearestKSearch(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
                            if (pointSearchSqDis_surf[NUM_MATCH_POINTS - 1] > 6.0)
                            {
                                point_selected_surf[i] =  false;
                            }
                        }
                        
                        if (! point_selected_surf[i]) continue;
                        /// using minimum square method
                        cv::Mat matA0(NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all(0));
                        cv::Mat matB0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(-1));
                        cv::Mat matX0(3, 1, CV_32F, cv::Scalar::all(0));

                        for (int j = 0; j < NUM_MATCH_POINTS; j++)
                        {
                            matA0.at<float>(j, 0) = featsFromMap->points[points_near[j]].x;
                            matA0.at<float>(j, 1) = featsFromMap->points[points_near[j]].y;
                            matA0.at<float>(j, 2) = featsFromMap->points[points_near[j]].z;
                        }
           
                        cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);  //TODO

                        float pa = matX0.at<float>(0, 0);
                        float pb = matX0.at<float>(1, 0);
                        float pc = matX0.at<float>(2, 0);
                        float pd = 1;

                        float ps = sqrt(pa * pa + pb * pb + pc * pc);
                        pa /= ps;
                        pb /= ps;
                        pc /= ps;
                        pd /= ps;

                        bool planeValid = true;
                        for (int j = 0; j < NUM_MATCH_POINTS; j++)
                        {
                            if (fabs(pa * featsFromMap->points[points_near[j]].x +
                                        pb * featsFromMap->points[points_near[j]].y +
                                        pc * featsFromMap->points[points_near[j]].z + pd) > 0.1)
                            {
                                planeValid = false;
                                point_selected_surf[i] = false;
                                break;
                            }
                        }

                        if (planeValid) 
                        {
                            float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                            float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y + pointSel_tmpt.z * pointSel_tmpt.z));

                            if (s > 0.92) 
                            {
                                if(std::abs(pd2) > 5 * res_mean_last)
                                {
                                    point_selected_surf[i] = false;
                                    res_last[i] = 0.0;
                                    continue;
                                }
                                point_selected_surf[i] = true;
                                coeffSel_tmpt->points[i].x = pa; 
                                coeffSel_tmpt->points[i].y = pb;
                                coeffSel_tmpt->points[i].z = pc;
                                coeffSel_tmpt->points[i].intensity = pd2;
                                res_last[i] = std::abs(pd2);
                            }
                            else
                            {
                                point_selected_surf[i] = false;
                            }
                        }

                    }

                    double total_residual = 0.0;
                    int laserCloudSelNum = 0;
                    for (int i = 0; i < coeffSel_tmpt->points.size(); i++)
                    {
                        if (point_selected_surf[i])
                        {
                            total_residual   += res_last[i];
                            laserCloudSelNum ++;
                        }
                    }

                    res_mean_last = total_residual / laserCloudSelNum;

                    for (int i = 0; i < coeffSel_tmpt->points.size(); i++)
                    {
                        float error_abs = std::abs(coeffSel_tmpt->points[i].intensity);
                        if (point_selected_surf[i])//  && (error_abs < 0.5))
                        {
                            laserCloudOri->push_back(feats_down->points[i]);
                            coeffSel->push_back(coeffSel_tmpt->points[i]);
                            total_residual += error_abs;
                        }
                    }

                    int state_dof = state::DOF;
                    if (laserCloudSelNum < 50) {
                        dyn_share.valid = false;
                        dyn_share.h_x = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, 23);
                        return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, 1);
                    }
                    
                    //prepare measurement model and its differentations 
                    Eigen::MatrixXd H(laserCloudSelNum, 23);
                    Eigen::VectorXd meas_vec(laserCloudSelNum);

                    omp_set_num_threads(4);
                    #pragma omp parallel for
                    for (int i = 0; i < laserCloudSelNum; i++)
                    {
                        const PointType &laser_p  = laserCloudOri->points[i];
                        Eigen::Vector3d point_this_be(laser_p.x, laser_p.y, laser_p.z);
                        Eigen::Matrix3d point_be_crossmat;
                        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
                        Eigen::Vector3d point_this = s.offset_R_L_I * point_this_be +s.offset_T_L_I;
                        Eigen::Matrix3d point_crossmat;
                        point_crossmat<<SKEW_SYM_MATRX(point_this);

                        /*** get the normal vector of closest surface/corner ***/
                        const PointType &norm_p = coeffSel->points[i];
                        Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

                        /*** calculate the Measuremnt Jacobian matrix H ***/
                        Eigen::Vector3d A(point_crossmat * s.rot.conjugate() * norm_vec);
                        Eigen::Vector3d B(point_be_crossmat * s.offset_R_L_I.conjugate() *s.rot.conjugate()*norm_vec);
                        Eigen::Vector3d C(s.rot.conjugate() *norm_vec);
                        H.row(i) = Eigen::Matrix<double, 1, 23>::Zero();
                        H.block<1, 3>(i,0) << norm_p.x, norm_p.y, norm_p.z;
                        H.block<1, 3>(i, 6) << VEC_FROM_ARRAY(A);
                        H.block<1, 3>(i, 17) << VEC_FROM_ARRAY(B);
                        H.block<1, 3>(1, 20) << VEC_FROM_ARRAY(C);

                        /*** Measuremnt: distance to the closest surface/corner ***/
                        meas_vec(i) = - norm_p.intensity;
                    }
                    dyn_share.h_x = H;
                    return meas_vec;
}
#else
Eigen::Matrix<double, Eigen::Dynamic, 1> get_h(state &s, esekfom::dyn_cal<double> &dyn_share)
{
                    
                    PointCloudXYZI::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
                    PointCloudXYZI::Ptr coeffSel(new pcl::PointCloud<PointType>());
                    int laserCloudSurf_down_size = feats_down->points.size();
                    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
                    PointCloudXYZI::Ptr coeffSel_tmpt(new pcl::PointCloud<PointType>(*feats_down));          
                    kdtreeSurfFromMap->setInputCloud(featsFromMap);
                    omp_set_num_threads(4);
                    #pragma omp parallel for
                    for (int i = 0; i < laserCloudSurf_down_size; i++)
                    {
                        PointType &pointOri_tmpt = feats_down->points[i];
                        PointType &pointSel_tmpt = feats_down_updated->points[i];
                        pointBodyToWorld(&pointOri_tmpt, &pointSel_tmpt, s);

                        std::vector<float> pointSearchSqDis_surf;
                        auto &points_near = pointSearchInd_surf[i];
                        if (dyn_share.converge)
                        {
                            point_selected_surf[i] = true;
                            kdtreeSurfFromMap->nearestKSearch(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
                            if (pointSearchSqDis_surf[NUM_MATCH_POINTS - 1] > 6.0)
                            {
                                point_selected_surf[i] = false;
                            }
                        }
                        
                        if (! point_selected_surf[i]) continue;

                        cv::Mat matA0(NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all(0));
                        cv::Mat matB0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(-1));
                        cv::Mat matX0(3, 1, CV_32F, cv::Scalar::all(0));

                        for (int j = 0; j < NUM_MATCH_POINTS; j++)
                        {
                            matA0.at<float>(j, 0) = featsFromMap->points[points_near[j]].x;
                            matA0.at<float>(j, 1) = featsFromMap->points[points_near[j]].y;
                            matA0.at<float>(j, 2) = featsFromMap->points[points_near[j]].z;
                        }
            
                        cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);  //TODO

                        float pa = matX0.at<float>(0, 0);
                        float pb = matX0.at<float>(1, 0);
                        float pc = matX0.at<float>(2, 0);
                        float pd = 1;

                        float ps = sqrt(pa * pa + pb * pb + pc * pc);
                        pa /= ps;
                        pb /= ps;
                        pc /= ps;
                        pd /= ps;

                        bool planeValid = true;
                        for (int j = 0; j < NUM_MATCH_POINTS; j++)
                        {
                            if (fabs(pa * featsFromMap->points[points_near[j]].x +
                                        pb * featsFromMap->points[points_near[j]].y +
                                        pc * featsFromMap->points[points_near[j]].z + pd) > 0.1)
                            {
                                planeValid = false;
                                point_selected_surf[i] = false;
                                break;
                            }
                        }

                        if (planeValid) 
                        {
                            float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                            float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y + pointSel_tmpt.z * pointSel_tmpt.z));

                            if (s > 0.92) 
                            {
                                if(std::abs(pd2) > 5 * res_mean_last)
                                {
                                    point_selected_surf[i] = false;
                                    res_last[i] = 0.0;
                                    continue;
                                }
                                point_selected_surf[i] = true;
                                coeffSel_tmpt->points[i].x = pa;
                                coeffSel_tmpt->points[i].y = pb;
                                coeffSel_tmpt->points[i].z = pc;
                                coeffSel_tmpt->points[i].intensity = pd2;
                                res_last[i] = std::abs(pd2);
                            }
                            else
                            {
                                point_selected_surf[i] = false;
                            }
                        }
                    }

                    double total_residual = 0.0;
                    int laserCloudSelNum = 0;
                    for (int i = 0; i < coeffSel_tmpt->points.size(); i++)
                    {
                        if (point_selected_surf[i])
                        {
                            total_residual   += res_last[i];
                            laserCloudSelNum ++;
                        }
                    }

                    res_mean_last = total_residual / laserCloudSelNum;

                    for (int i = 0; i < coeffSel_tmpt->points.size(); i++)
                    {
                        float error_abs = std::abs(coeffSel_tmpt->points[i].intensity);
                        if (point_selected_surf[i])
                        {
                            laserCloudOri->push_back(feats_down->points[i]);
                            coeffSel->push_back(coeffSel_tmpt->points[i]);
                            total_residual += error_abs;
                        }
                    }

                    int state_dof = state::DOF;
                    if (laserCloudSelNum < 50) {
                        dyn_share.valid = false;
                        dyn_share.h_x = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, state_dof);
                        dyn_share.z = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, 1);
                        dyn_share.h_v = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, laserCloudSelNum);
                        dyn_share.R = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, laserCloudSelNum);
                        return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, 1);
                    }
                    
                    // prepare measurement model and its differentations 
                    Eigen::MatrixXd H(laserCloudSelNum, state_dof);
                    Eigen::VectorXd meas_vec(laserCloudSelNum);

                    omp_set_num_threads(4);
                    #pragma omp parallel for
                    for (int i = 0; i < laserCloudSelNum; i++)
                    {
                        const PointType &laser_p  = laserCloudOri->points[i];
                        Eigen::Vector3d point_this_be(laser_p.x, laser_p.y, laser_p.z);
                        Eigen::Matrix3d point_be_crossmat;
                        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
                        Eigen::Vector3d point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
                        Eigen::Matrix3d point_crossmat;
                        point_crossmat<<SKEW_SYM_MATRX(point_this);

                        /*** get the normal vector of closest surface/corner ***/
                        const PointType &norm_p = coeffSel->points[i];
                        Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

                        /*** calculate the Measuremnt Jacobian matrix H ***/
                        Eigen::Vector3d A(point_crossmat * s.rot.conjugate() * norm_vec);
                        Eigen::Vector3d B(point_be_crossmat * s.offset_R_L_I.conjugate() *s.rot.conjugate()*norm_vec);
                        Eigen::Vector3d C(s.rot.conjugate() *norm_vec);
                        H.row(i) = Eigen::Matrix<double, 1, 23>::Zero();
                        H.block<1, 3>(i, 0) << norm_p.x, norm_p.y, norm_p.z;
                        H.block<1, 3>(i, 6) << VEC_FROM_ARRAY(A);
                        H.block<1, 3>(i, 17) << VEC_FROM_ARRAY(B);
                        H.block<1, 3>(1, 20) << VEC_FROM_ARRAY(C);

                        /*** Measuremnt: distance to the closest surface/corner ***/
                        meas_vec(i) = norm_p.intensity;
                    }

                    dyn_share.h_v = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(laserCloudSelNum, laserCloudSelNum);
                    dyn_share.R = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(laserCloudSelNum, laserCloudSelNum) * 0.03;
                    dyn_share.h_x = H; //= meas_vec;
                    dyn_share.z = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, 1);
                    return meas_vec;
}
#endif
void SigHandle(int sig)
{
  b_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}



void RGBpointBodyToWorld(PointType const * const pi, pcl::PointXYZRGB * const po, state &s)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);

    float intensity = pi->intensity;
    intensity = intensity - std::floor(intensity);

    int reflection_map = intensity*10000;

    if (reflection_map < 30)
    {
        int green = (reflection_map * 255 / 30);
        po->r = 0;
        po->g = green & 0xff;
        po->b = 0xff;
    }
    else if (reflection_map < 90)
    {
        int blue = (((90 - reflection_map) * 255) / 60);
        po->r = 0x0;
        po->g = 0xff;
        po->b = blue & 0xff;
    }
    else if (reflection_map < 150)
    {
        int red = ((reflection_map-90) * 255 / 60);
        po->r = red & 0xff;
        po->g = 0xff;
        po->b = 0x0;
    }
    else
    {
        int green = (((255-reflection_map) * 255) / (255-150));
        po->r = 0xff;
        po->g = green & 0xff;
        po->b = 0;
    }
}

void lasermap_fov_segment(state &s)
{
    laserCloudValidNum    = 0;
    laserCloudSurroundNum = 0;
    PointType pointOnYAxis;
    pointOnYAxis.x = LIDAR_SP_LEN;
    pointOnYAxis.y = 0.0;
    pointOnYAxis.z = 0.0;
    pointBodyToWorld(&pointOnYAxis, &pointOnYAxis, s);
    Eigen::Vector3d pos_LiD = s.rot*s.offset_T_L_I + s.pos; // s.pos; 
    int centerCubeI = int((pos_LiD[0] + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
    int centerCubeJ = int((pos_LiD[1] + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
    int centerCubeK = int((pos_LiD[2] + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

    if (pos_LiD[0] + 0.5 * cube_len < 0) centerCubeI--;
    if (pos_LiD[1] + 0.5 * cube_len < 0) centerCubeJ--;
    if (pos_LiD[2] + 0.5 * cube_len < 0) centerCubeK--;

    while (centerCubeI < 3)
    {
        for (int j = 0; j < laserCloudHeight; j++)
        {
            for (int k = 0; k < laserCloudDepth; k++)
            {
                int i = laserCloudWidth - 1;

                PointCloudXYZI::Ptr laserCloudCubeSurfPointer =
                        featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                for (; i >= 1; i--) {
                    featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            featsArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                }
                featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeI++;
        laserCloudCenWidth++;
    }

    while (centerCubeI >= laserCloudWidth - 3) {
        for (int j = 0; j < laserCloudHeight; j++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int i = 0;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer =
                        featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                for (; i < laserCloudWidth - 1; i++)
                {
                    featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            featsArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                }

                featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeI--;
        laserCloudCenWidth--;
    }

    while (centerCubeJ < 3) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int j = laserCloudHeight - 1;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer =
                        featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                for (; j >= 1; j--) {
                    featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            featsArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight*k];
                }
                featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
    }

    while (centerCubeJ >= laserCloudHeight - 3) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int j = 0;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer =
                        featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                for (; j < laserCloudHeight - 1; j++) {
                    featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            featsArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight*k];
                }
                featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
    }

    while (centerCubeK < 3) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int j = 0; j < laserCloudHeight; j++) {
                int k = laserCloudDepth - 1;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer =
                        featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                for (; k >= 1; k--) {
                    featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight*(k - 1)];
                }
                featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK++;
        laserCloudCenDepth++;
    }

    while (centerCubeK >= laserCloudDepth - 3)
    {
        for (int i = 0; i < laserCloudWidth; i++)
        {
            for (int j = 0; j < laserCloudHeight; j++)
            {
                int k = 0;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer =
                        featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            
                for (; k < laserCloudDepth - 1; k++)
                {
                    featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                            featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight*(k + 1)];
                }

                featsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        laserCloudCubeSurfPointer;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK--;
        laserCloudCenDepth--;
    }

    for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) 
    {
        for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) 
        {
            for (int k = centerCubeK - 2; k <= centerCubeK + 2; k++) 
            {
                if (i >= 0 && i < laserCloudWidth &&
                        j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth) 
                {

                    float centerX = cube_len * (i - laserCloudCenWidth);
                    float centerY = cube_len * (j - laserCloudCenHeight);
                    float centerZ = cube_len * (k - laserCloudCenDepth);

                    float check1, check2;
                    float squaredSide1, squaredSide2;
                    float ang_cos = 1;

                    bool isInLaserFOV = false;

                    for (int ii = -1; ii <= 1; ii += 2) 
                    {
                        for (int jj = -1; jj <= 1; jj += 2) 
                        {
                            for (int kk = -1; (kk <= 1) && (!isInLaserFOV); kk += 2) 
                            {

                                float cornerX = centerX + 0.5 * cube_len * ii;
                                float cornerY = centerY + 0.5 * cube_len * jj;
                                float cornerZ = centerZ + 0.5 * cube_len * kk;

                                squaredSide1 = (pos_LiD[0] - cornerX)
                                        * (pos_LiD[0] - cornerX)
                                        + (pos_LiD[1] - cornerY)
                                        * (pos_LiD[1] - cornerY)
                                        + (pos_LiD[2] - cornerZ)
                                        * (pos_LiD[2] - cornerZ);

                                squaredSide2 = (pointOnYAxis.x - cornerX) * (pointOnYAxis.x - cornerX)
                                        + (pointOnYAxis.y - cornerY) * (pointOnYAxis.y - cornerY)
                                        + (pointOnYAxis.z - cornerZ) * (pointOnYAxis.z - cornerZ);

                                ang_cos = fabs(squaredSide1 <= 3) ? 1.0 :
                                    (LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2) / (2 * LIDAR_SP_LEN * sqrt(squaredSide1));
                                
                                if(ang_cos > HALF_FOV_COS) isInLaserFOV = true;
                            }
                        }
                    }
                    
                    if(!isInLaserFOV)
                    {
                        float cornerX = centerX;
                        float cornerY = centerY;
                        float cornerZ = centerZ;

                        squaredSide1 = (pos_LiD[0] - cornerX)
                                * (pos_LiD[0] - cornerX)
                                + (pos_LiD[1] - cornerY)
                                * (pos_LiD[1] - cornerY)
                                + (pos_LiD[2] - cornerZ)
                                * (pos_LiD[2] - cornerZ);

                        if(squaredSide1 <= 0.4 * cube_len * cube_len)
                        {
                            isInLaserFOV = true;
                        }

                        squaredSide2 = (pointOnYAxis.x - cornerX) * (pointOnYAxis.x - cornerX)
                                + (pointOnYAxis.y - cornerY) * (pointOnYAxis.y - cornerY)
                                + (pointOnYAxis.z - cornerZ) * (pointOnYAxis.z - cornerZ);
                        
                        ang_cos = fabs(squaredSide2 <= 0.5 * cube_len) ? 1.0 :
                            (LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2) / (2 * LIDAR_SP_LEN * sqrt(squaredSide1));
                        
                        if(ang_cos > HALF_FOV_COS) isInLaserFOV = true;
                    }

                    
                    if (isInLaserFOV)
                    {
                        laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j
                                + laserCloudWidth * laserCloudHeight * k;
                        laserCloudValidNum ++;
                    }
                    laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j
                            + laserCloudWidth * laserCloudHeight * k;
                    laserCloudSurroundNum ++;

                }
            }
        }
    }

    featsFromMap->clear();
    
    for (int i = 0; i < laserCloudValidNum; i++)
    {
        *featsFromMap += *featsArray[laserCloudValidInd[i]];
    }
}

void feat_points_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    lidar_buffer.push_back(msg);
    last_timestamp_lidar = msg->header.stamp.toSec();

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        b_reset = true;
        b_first = true;
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push lidar frame ***/
    if(!lidar_pushed)
    {
        meas.lidar.reset(new PointCloudXYZI());
        pcl::fromROSMsg(*(lidar_buffer.front()), *(meas.lidar));
        meas.lidar_beg_time = lidar_buffer.front()->header.stamp.toSec();
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    ros::Subscriber sub_pcl = nh.subscribe("/laser_cloud_flat", 20000, feat_points_cbk);
    ros::Subscriber sub_imu = nh.subscribe("/livox/imu", 20000, imu_cbk);
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100);
    ros::Publisher pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/aft_mapped_to_init", 10);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 10);
#ifdef DEPLOY
    ros::Publisher mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
#endif
    geometry_msgs::PoseStamped msg_body_pose;
    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="/camera_init";

    double limit[23] = {10000.0};
    limit[6] = 0.0027;
    limit[7] = 0.0027;
    limit[8] = 0.0027;
    limit[0] = 0.0015;
    limit[1] = 0.0015;
    limit[2] = 0.0015;
    /*** variables definition ***/
    bool dense_map_en, Need_Init = true;
    std::string map_file_path;
    int effect_feat_num = 0, frame_num = 0;
    double filter_size_corner_min, filter_size_surf_min, filter_size_map_min, fov_deg,\
           deltaT, deltaR, aver_time_consu = 0, first_lidar_time = 0;
    double aver_time_consu_predict = 0;



    nav_msgs::Odometry odomAftMapped;

    cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

    PointType pointOri, pointSel, coeff;
    PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
    PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
    PointCloudXYZI::Ptr coeffSel(new PointCloudXYZI());
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    /*** variables initialize ***/
    ros::param::get("~dense_map_enable",dense_map_en);
    ros::param::get("~max_iteration",NUM_MAX_ITERATIONS);
    ros::param::get("~map_file_path",map_file_path);
    ros::param::get("~fov_degree",fov_deg);
    ros::param::get("~filter_size_corner",filter_size_corner_min);
    ros::param::get("~filter_size_surf",filter_size_surf_min);
    ros::param::get("~filter_size_map",filter_size_map_min);
    ros::param::get("~cube_side_length",cube_len);

// initialize the Kalman filter
#ifdef Modified
    state_cal.kf.init_dyn_share(get_f, df_dx, df_dw, get_h_modified, NUM_MAX_ITERATIONS, limit);
#else
    state_cal.kf.init_dyn_share(get_f, df_dx, df_dw, get_h, NUM_MAX_ITERATIONS, limit);
#endif    
    HALF_FOV_COS = std::cos((fov_deg + 10.0) * 0.5 * PI_M / 180.0);

    for (int i = 0; i < laserCloudNum; i++)
    {
        featsArray[i].reset(new PointCloudXYZI());
    }

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    std::shared_ptr<ImuProcess> p_imu(new ImuProcess());

    /*** debug record ***/
    std::ofstream fout_pre, fout_out;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),std::ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),std::ios::out);
    if (fout_pre && fout_out)
        std::cout << "~~~~"<<ROOT_DIR<<" file opened" << std::endl;
    else
        std::cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << std::endl;
    
    std::vector<double> T1, s_plot, s_plot_predict;
    std::vector<vect3> s_plot2, s_plot3, s_plot4, s_plot5, s_plot6, s_plot7, s_plot8, s_plot9, s_plot_e1, s_plot_e2, s_plot_e3, s_plot_e4, s_plot_e5, s_plot_e6, s_plot_e7, s_plot_e8, s_plot_e9, s_plot_e10, s_plot_e11, s_plot_e12, s_plot_e13, s_plot_e14;
    std::vector<vect2> s_plot_e15, s_plot_e16;
    vect3 pos_cov, rot_cov, vel_cov, gra_cov, offr_cov, offt_cov, bg_cov, ba_cov, gra_plus, gra_minus, gra_plus1, gra_minus1, gra_plus2, gra_minus2;
    
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (b_exit) break;
        ros::spinOnce();
        while(sync_packages(Measures)) 
        {
            if (b_reset)
            {
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();
                b_reset = false;
                continue;
            }

            double t0,t1,t2,t3,t4, t5,match_start, match_time, solve_start, solve_time, pca_time, svd_time;
            match_time = 0;
            solve_time = 0;
            pca_time   = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, state_cal, feats_undistort);

            t5 = omp_get_wtime();
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                first_lidar_time = Measures.lidar_beg_time;
                std::cout<<"not ready for odometry"<<std::endl;
                continue;
            }

            if ((Measures.lidar_beg_time - first_lidar_time) < INIT_TIME)
            {
                Need_Init = true;
                std::cout<<"||||||||||Initiallizing LiDar||||||||||"<<std::endl;
            }
            else
            {
                Need_Init = false;
            }
            
            state cal_state = state_cal.kf.get_x();
            /*** Compute the euler angle ***/
            Eigen::Vector3d euler_cur = RotMtoEuler(cal_state.rot.toRotationMatrix());
            fout_pre << std::setw(10) << Measures.lidar_beg_time << " " << euler_cur.transpose()*57.3 << " " << cal_state.pos.transpose() << " " << cal_state.vel.transpose() \
            <<" "<<cal_state.bg.transpose()<<" "<<cal_state.ba.transpose()<< std::endl;
            #ifdef DEBUG_PRINT
            std::cout<<"current lidar time "<<Measures.lidar_beg_time<<" "<<"first lidar time "<<first_lidar_time<<std::endl;
            #endif
            
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment(cal_state);
            
            /*** downsample the features and maps ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down);
            downSizeFilterMap.setInputCloud(featsFromMap);
            downSizeFilterMap.filter(*featsFromMap);

            int featsFromMapNum = featsFromMap->points.size();
            int feats_down_size = feats_down->points.size();
            std::cout<<"[ mapping ]: Raw feature num: "<<feats_undistort->points.size()<<" downsamp num "<<feats_down_size<<" Map num: "<<featsFromMapNum<<" laserCloudValidNum "<<laserCloudValidNum<<std::endl;

            /*** ICP and iterated Kalman filter update ***/
            PointCloudXYZI::Ptr coeffSel_tmpt (new PointCloudXYZI(*feats_down));
            *feats_down_updated = *feats_down;

            if (featsFromMapNum > 100)
            {
                t1 = omp_get_wtime();
                kdtreeSurfFromMap->setInputCloud(featsFromMap);
                point_selected_surf.clear();
                res_last.clear();
                for(int i = 0; i < feats_down_size; i++)
                {
                    point_selected_surf.push_back(true);
                    res_last.push_back(1000.0);
                }
                pointSearchInd_surf.clear();
                pointSearchInd_surf.resize(feats_down_size);
                if(Need_Init)
                {
                    std::cout << "do not need initialize!" << std::endl;
                }
                else
                {
                    t2 = omp_get_wtime();
                #ifdef Modified
                    state_cal.kf.update_iterated_dyn_share_modified(); //, NUM_MAX_ITERATIONS, limit, 100.0);
                #else
                    state_cal.kf.update_iterated_dyn_share(); //NUM_MAX_ITERATIONS, limit, 100.0);   
                #endif                    
                    t3 = omp_get_wtime();
                    state for_transform = state_cal.kf.get_x();
                    euler_cur = RotMtoEuler(for_transform.rot.toRotationMatrix());

                    omp_set_num_threads(4);
                    #pragma omp parallel for
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        PointType &pointOri_tmpt = feats_down->points[i];
                        PointType &pointSel_tmpt = feats_down_updated->points[i];
                        pointBodyToWorld(&pointOri_tmpt, &pointSel_tmpt, for_transform);
                    }
                    
                }
            }
            bool if_cube_updated[laserCloudNum] = {0};
            for (int i = 0; i < feats_down_size; i++)
            {
                PointType &pointSel = feats_down_updated->points[i];

                int cubeI = int((pointSel.x + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
                int cubeJ = int((pointSel.y + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
                int cubeK = int((pointSel.z + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

                if (pointSel.x + 0.5 * cube_len < 0) cubeI--;
                if (pointSel.y + 0.5 * cube_len < 0) cubeJ--;
                if (pointSel.z + 0.5 * cube_len < 0) cubeK--;

                if (cubeI >= 0 && cubeI < laserCloudWidth &&
                        cubeJ >= 0 && cubeJ < laserCloudHeight &&
                        cubeK >= 0 && cubeK < laserCloudDepth) {
                    int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                    featsArray[cubeInd]->push_back(pointSel);
                    if_cube_updated[cubeInd] = true;
                }
            }
            for (int i = 0; i < laserCloudValidNum; i++)
            {
                int ind = laserCloudValidInd[i];

                if(if_cube_updated[ind])
                {
                    downSizeFilterMap.setInputCloud(featsArray[ind]);
                    downSizeFilterMap.filter(*featsArray[ind]);
                }
            }

            state for_save = state_cal.kf.get_x();
            /******* Publish current frame points in world coordinates:  *******/
            laserCloudFullRes2->clear();
            *laserCloudFullRes2 = dense_map_en ? (*feats_undistort) : (* feats_down);

            int laserCloudFullResNum = laserCloudFullRes2->points.size();
    
            pcl::PointXYZRGB temp_point;
            laserCloudFullResColor->clear();
            {
            for (int i = 0; i < laserCloudFullResNum; i++)
            {
                RGBpointBodyToWorld(&laserCloudFullRes2->points[i], &temp_point, for_save);
                laserCloudFullResColor->push_back(temp_point);
            }

            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
            laserCloudFullRes3.header.stamp = ros::Time::now();
            laserCloudFullRes3.header.frame_id = "/camera_init";
            pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }

            /******* Publish Effective points *******/
            {
            laserCloudFullResColor->clear();
            pcl::PointXYZRGB temp_point;
            for (int i = 0; i < laserCloudSelNum; i++)
            {
                RGBpointBodyToWorld(&laserCloudOri->points[i], &temp_point, for_save);
                laserCloudFullResColor->push_back(temp_point);
            }
            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
            laserCloudFullRes3.header.stamp = ros::Time::now();
            laserCloudFullRes3.header.frame_id = "/camera_init";
            pubLaserCloudEffect.publish(laserCloudFullRes3);
            }

            /******* Publish Maps:  *******/
            sensor_msgs::PointCloud2 laserCloudMap;
            pcl::toROSMsg(*featsFromMap, laserCloudMap);
            laserCloudMap.header.stamp = ros::Time::now();
            laserCloudMap.header.frame_id = "/camera_init";
            pubLaserCloudMap.publish(laserCloudMap);

            /******* Publish Odometry ******/
            odomAftMapped.header.frame_id = "/camera_init";
            odomAftMapped.child_frame_id = "/aft_mapped";

            odomAftMapped.header.stamp = ros::Time::now();
            odomAftMapped.pose.pose.orientation.x = for_save.rot.x(); 
            odomAftMapped.pose.pose.orientation.y = for_save.rot.y(); 
            odomAftMapped.pose.pose.orientation.z = for_save.rot.z(); 
            odomAftMapped.pose.pose.orientation.w = for_save.rot.w(); 
            odomAftMapped.pose.pose.position.x = for_save.pos[0];
            odomAftMapped.pose.pose.position.y = for_save.pos[1];
            odomAftMapped.pose.pose.position.z = for_save.pos[2];

            pubOdomAftMapped.publish(odomAftMapped);

            static tf::TransformBroadcaster br;
            tf::Transform                   transform;
            tf::Quaternion                  q;
            transform.setOrigin( tf::Vector3( odomAftMapped.pose.pose.position.x,
                                                odomAftMapped.pose.pose.position.y,
                                                odomAftMapped.pose.pose.position.z ) );
            q.setW( odomAftMapped.pose.pose.orientation.w );
            q.setX( odomAftMapped.pose.pose.orientation.x );
            q.setY( odomAftMapped.pose.pose.orientation.y );
            q.setZ( odomAftMapped.pose.pose.orientation.z );
            transform.setRotation( q );
            br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped" ) );

            
            msg_body_pose.header.stamp = ros::Time::now();
            msg_body_pose.header.frame_id = "/camera_odom_frame";
            msg_body_pose.pose.position.x = for_save.pos[0];
            msg_body_pose.pose.position.y = for_save.pos[1];
            msg_body_pose.pose.position.z = for_save.pos[2];
            msg_body_pose.pose.orientation.x = for_save.rot.x(); 
            msg_body_pose.pose.orientation.y = for_save.rot.y(); 
            msg_body_pose.pose.orientation.z = for_save.rot.z(); 
            msg_body_pose.pose.orientation.w = for_save.rot.w(); 
            #ifdef DEPLOY
            mavros_pose_publisher.publish(msg_body_pose);
            #endif

            /******* Publish Path ********/
            msg_body_pose.header.frame_id = "/camera_init";
            path.poses.push_back(msg_body_pose);
            pubPath.publish(path);

            /*** save debug variables ***/
            t4 = omp_get_wtime();
            frame_num ++;
            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t3 - t2) / frame_num;
            aver_time_consu_predict = aver_time_consu_predict * (frame_num - 1) / frame_num + (t5- t0) / frame_num;
            std::cout << "update_time:" << aver_time_consu << " s" << "|||" << "predict_time:" << aver_time_consu_predict << " s" << std::endl;

            T1.push_back(Measures.lidar_beg_time);
            s_plot.push_back(aver_time_consu);
            s_plot_predict.push_back(aver_time_consu_predict);
            s_plot2.push_back(for_save.offset_T_L_I);
            SO3 off_r_save = for_save.offset_R_L_I;
            vect3 euler_save =  SO3ToEuler(off_r_save);
            s_plot3.push_back(euler_save);
            s_plot4.push_back(for_save.vel);
            SO3 r_save = for_save.rot;
            vect3 euler_save_r = SO3ToEuler(r_save);
            s_plot5.push_back(euler_save_r);
            s_plot6.push_back(for_save.pos);
            s_plot7.push_back(for_save.bg);
            s_plot8.push_back(for_save.ba);
            s_plot9.push_back(for_save.grav);
            esekfom::esekf<state, 12, input>::cov cov_stat_cur = state_cal.kf.get_P();
            double std_grav[2] = {std::sqrt(cov_stat_cur(15, 15)), std::sqrt(cov_stat_cur(16, 16))};
            double std_gravp[2] = {3*std::sqrt(cov_stat_cur(15, 15)), 3*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravm[2] = {-3*std::sqrt(cov_stat_cur(15, 15)), -3*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravp1[2] = {3*std::sqrt(cov_stat_cur(15, 15)), 0*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravp2[2] = {0*std::sqrt(cov_stat_cur(15, 15)), 3*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravm1[2] = {-3*std::sqrt(cov_stat_cur(15, 15)), 0*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravm2[2] = {0*std::sqrt(cov_stat_cur(15, 15)), -3*std::sqrt(cov_stat_cur(16, 16))};
		    vect2 delta(std_grav, 2);
            vect2 delta_cur;
            S2 grav_con(0.042761, 0.153072, -9.80771);
            grav_con.boxminus(delta_cur, for_save.grav);
		    vect2 deltap(std_gravp, 2);
		    vect2 deltap1(std_gravp1, 2);
		    vect2 deltap2(std_gravp2, 2);
		    vect2 deltam(std_gravm, 2);
		    vect2 deltam1(std_gravm1, 2);
		    vect2 deltam2(std_gravm2, 2);
            S2 grav_plus(for_save.grav);
            S2 grav_plus1(for_save.grav);
            S2 grav_plus2(for_save.grav);
            S2 grav_minus(for_save.grav);
            S2 grav_minus1(for_save.grav);
            S2 grav_minus2(for_save.grav);
            grav_plus.boxplus(deltap);
            grav_plus1.boxplus(deltap1);
            grav_plus2.boxplus(deltap2);
            grav_minus.boxplus(deltam);
            grav_minus1.boxplus(deltam1);
            grav_minus2.boxplus(deltam2);
		    Eigen::Matrix<state::scalar, 3, 2> grav_matrix;
			MTK::vect<2, state::scalar> vec = MTK::vect<2, state::scalar>::Zero();
		    for_save.S2_Mx(grav_matrix, vec, 15);
            Eigen::Matrix<double, 3, 3> JcovJT = grav_matrix * cov_stat_cur.template block<2, 2>(15, 15) * grav_matrix.transpose();
		    for(int i=0; i<3; i++)
		    {
			    pos_cov(i) = std::sqrt(cov_stat_cur(i, i));
			    rot_cov(i) = std::sqrt(cov_stat_cur(i+6, i+6));
			    vel_cov(i) = std::sqrt(cov_stat_cur(3+i, 3+i));
                bg_cov(i) = std::sqrt(cov_stat_cur(9+i, 9+i));
                ba_cov(i) = std::sqrt(cov_stat_cur(12+i, 12+i));
			    offr_cov(i) = std::sqrt(cov_stat_cur(17+i, 17+i));
			    offt_cov(i) = std::sqrt(cov_stat_cur(20+i, 20+i));
			    gra_cov(i) = std::sqrt(JcovJT(i, i)); 
                gra_plus(i) = grav_plus[i];
                gra_plus1(i) = grav_plus1[i];
                gra_plus2(i) = grav_plus2[i];
                gra_minus(i) = grav_minus[i];
                gra_minus1(i) = grav_minus1[i];
                gra_minus2(i) = grav_minus2[i];
		    }
		    s_plot_e1.push_back(pos_cov);
		    s_plot_e3.push_back(rot_cov);
		    s_plot_e2.push_back(vel_cov);
		    s_plot_e4.push_back(gra_cov);
		    s_plot_e5.push_back(offr_cov);
		    s_plot_e6.push_back(offt_cov);
            s_plot_e7.push_back(bg_cov);
            s_plot_e8.push_back(ba_cov);
            s_plot_e9.push_back(gra_plus);
            s_plot_e11.push_back(gra_plus1);
            s_plot_e13.push_back(gra_plus2);
            s_plot_e10.push_back(gra_minus);
            s_plot_e12.push_back(gra_minus1);
            s_plot_e14.push_back(gra_minus2);
            s_plot_e15.push_back(delta_cur);
            s_plot_e16.push_back(delta);


            std::cout<<"[ mapping ]: time: segm "<<t1-t0 <<" kdtree build: "<<t2-t1<<" match "<<match_time<<" pca "<<pca_time<<" solve "<<solve_time<<" total "<<t4 - t0<<std::endl;
            fout_out << std::setw(10) << Measures.lidar_beg_time << " " << euler_cur.transpose()*57.3 << " " << for_save.pos.transpose() << " " << for_save.vel.transpose() \
            <<" "<<for_save.bg.transpose()<<" "<<for_save.ba.transpose()<< std::endl;
        }
        status = ros::ok();
        rate.sleep();
    }
    //--------------------------save map---------------
    std::string surf_filename(map_file_path + "/surf.pcd");
    std::string corner_filename(map_file_path + "/corner.pcd");
    std::string all_points_filename(map_file_path + "/all_points.pcd");

    PointCloudXYZI surf_points, corner_points;
    surf_points = *featsFromMap;
    fout_out.close();
    fout_pre.close();
    if (surf_points.size() > 0 && corner_points.size() > 0) 
    {
    pcl::PCDWriter pcd_writer;
    std::cout << "saving...";
    pcd_writer.writeBinary(surf_filename, surf_points);
    pcd_writer.writeBinary(corner_filename, corner_points);
    }
    else
    {
        // #ifdef DEBUG_PRINT
        #ifndef DEPLOY
        if (!T1.empty())
        {
            std::string  save_off = "/home/ubuntu/fast_lio4/estimate_off.csv";
            std::string save_cov="/home/ubuntu/fast_lio4/estimate_cov.csv";
            std::string save_time="/home/ubuntu/fast_lio4/consumed_time.csv";
            outputData(save_off, T1, s_plot3, s_plot2, s_plot4, s_plot5, s_plot6, s_plot7, s_plot8, s_plot9);
            outputCov(save_cov, s_plot_e1, s_plot_e2, s_plot_e3, s_plot_e4, s_plot_e5, s_plot_e6, s_plot_e7, s_plot_e8, s_plot_e9, s_plot_e10, s_plot_e11, s_plot_e12, s_plot_e13, s_plot_e14, s_plot_e15, s_plot_e16);
            outputAvetime(save_time, s_plot, s_plot_predict);
            plt::named_plot("time consumed",T1,s_plot);
            plt::legend();
            plt::show();
            plt::pause(0.5);
            plt::close();
        }
        std::cout << "no points saved";
        #endif
    }
    return 0;
}
