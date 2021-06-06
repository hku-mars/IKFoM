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
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>

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


#ifdef DEPLOY
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

#define INIT_TIME           (0)
#define LASER_POINT_COV     (0.0015) //0.0015
#define NUM_MATCH_POINTS    (5)

std::string root_dir = ROOT_DIR;

int iterCount = 0;
int NUM_MAX_ITERATIONS  = 0;
int FOV_RANGE = 3;  // range of FOV = FOV_RANGE * cube_len
int laserCloudCenWidth  = 24;
int laserCloudCenHeight = 24;
int laserCloudCenDepth  = 24;

int laserCloudValidNum    = 0;
int laserCloudSelNum      = 0;

const int laserCloudWidth  = 48;
const int laserCloudHeight = 48;
const int laserCloudDepth  = 48;
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;

// std::vector<double> T1, T2, s_plot, s_plot2, s_plot3, s_plot4, s_plot5, s_plot6;

std::vector<double> T1, T2, s_averge_time, s_total_time;
#ifdef USE_IKFOM
std::vector<vect3> s_offt, s_offr, s_vel, s_pos, s_rot, s_bg, s_ba, s_grav, s_cov_pos, s_cov_vel, s_cov_rot_plus, s_cov_rot_minus, s_cov_grav, s_cov_offr_plus, s_cov_offr_minus, s_cov_offt, s_cov_bg, s_cov_ba, s_grav_plus, s_grav_minus, s_grav_plus1, s_grav_minus1, s_grav_plus2, s_grav_minus2;
std::vector<vect2> s_grav_deltacur, s_grav_delta;
vect3 pos_cov, rot_cov, vel_cov, gra_cov, offr_cov, offt_cov, bg_cov, ba_cov, gra_plus, gra_minus, gra_plus1, gra_minus1, gra_plus2, gra_minus2;
#else
std::vector<Eigen::Vector3d> s_offt, s_offr, s_vel, s_pos, s_rot, s_bg, s_ba, s_grav, s_cov_pos, s_cov_vel, s_cov_rot_plus, s_cov_rot_minus, s_cov_grav, s_cov_offr, s_cov_offt, s_cov_bg, s_cov_ba, s_grav_plus, s_grav_minus, s_grav_plus1, s_grav_minus1, s_grav_plus2, s_grav_minus2;
Eigen::Vector3d pos_cov, rot_cov, vel_cov, gra_cov, offr_cov, offt_cov, bg_cov, ba_cov, gra_plus, gra_minus, gra_plus1, gra_minus1, gra_plus2, gra_minus2;
#endif



#ifdef USE_IKFOM
#ifdef RESTORE_VICON
// std::vector<Eigen::Vector3d> pose_g, vel_g;
std::vector<vect3> pose_g, vel_g, rot_g;
// std::vector<SO3> rot_g;
// Eigen::Vector3d v_pose, v_vel;
vect3 v_pose, v_vel, v_rot, init_pose;
// SO3 v_rot;
SO3 init_rot;
std::vector<double> v_time;
int vicon_first = 0;
#endif
#endif

/// IMU relative variables
std::mutex mtx_buffer;
std::condition_variable sig_buffer;
bool lidar_pushed = false;
bool flg_exit = false;
bool flg_reset = false;
bool flg_map_inited = false;

/// Buffers for measurements
double cube_len = 0.0;
double lidar_end_time = 0.0;
double last_timestamp_lidar = -1;
double last_timestamp_imu   = -1;
double last_timestamp_vicon_pose   = -1;
double last_timestamp_vicon_vel   = -1;
double HALF_FOV_COS = 0.0;
double res_mean_last = 0.05;
double total_distance = 0.0;
auto   position_last  = Zero3d;
double copy_time, readd_time;

double total_residual = 0.0;
int feats_down_size = 0;
#ifdef USE_ikdtree
    std::vector<PointVector> Nearest_Points;
#endif
std::vector<std::vector<int>> pointSearchInd_surf;
std::deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::vector<bool> point_selected_surf; 
std::vector<double> res_last;
Eigen::Vector3d euler_cur;
geometry_msgs::Quaternion geoQuat;

//surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr cube_points_add(new PointCloudXYZI());

PointCloudXYZI::Ptr feats_down(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_updated(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr coeffSel(new PointCloudXYZI());
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr coeffSel_tmpt (new PointCloudXYZI(100000, 1));
//all points
PointCloudXYZI::Ptr laserCloudFullRes2(new PointCloudXYZI());
PointCloudXYZI::Ptr featsArray[laserCloudNum];
bool                _last_inFOV[laserCloudNum];
bool                cube_updated[laserCloudNum];
int laserCloudValidInd[laserCloudNum];
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullResColor(new pcl::PointCloud<pcl::PointXYZI>());

#ifdef USE_ikdtree
KD_TREE ikdtree;
std::vector<BoxPointType> cub_needrm;
std::vector<BoxPointType> cub_needad;
#else
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
#endif

Eigen::Vector3f XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
Eigen::Vector3f XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);

//estimator inputs and output;
MeasureGroup Measures;
#ifdef USE_IKFOM
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;
#else
StatesGroup  state;
#endif

void SigHandle(int sig)
{
  flg_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}

//project lidar frame to world
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    #ifdef USE_IKFOM
    #ifdef Online_Calibration
    Eigen::Vector3d p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    #else
    Eigen::Vector3d p_global(state_point.rot * (p_body + Lid_offset_to_IMU) + state_point.pos);
    #endif
    #else
    #ifdef USE_QUA
    Eigen::Matrix3d rot_qua;
    rot_qua<<ROT_FROM_QUA(state.qua_end);
    #ifdef ON_CAL_QUA
    Eigen::Matrix3d rot_off;
    rot_off<<ROT_FROM_QUA(state.off_r);
    Eigen::Vector3d p_global(rot_qua * (rot_off*p_body + state.off_t) + state.pos_end);
    #else
    Eigen::Vector3d p_global(rot_qua * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif
    #else
    Eigen::Vector3d p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif
    #endif

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po)
{
    Eigen::Vector3d p_body(pi[0], pi[1], pi[2]);
    #ifdef USE_IKFOM
    #ifdef Online_Calibration
    Eigen::Vector3d p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    #else
    Eigen::Vector3d p_global(state_point.rot * (p_body + Lid_offset_to_IMU) + state_point.pos);
    #endif
    #else
    #ifdef USE_QUA
    Eigen::Matrix3d rot_qua;
    rot_qua<<ROT_FROM_QUA(state.qua_end);
    #ifdef ON_CAL_QUA
    Eigen::Matrix3d rot_off;
    rot_off<<ROT_FROM_QUA(state.off_r);
    Eigen::Vector3d p_global(rot_qua * (rot_off*p_body + state.off_t) + state.pos_end);
    #else
    Eigen::Vector3d p_global(rot_qua * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif
    #else
    Eigen::Vector3d p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif
    #endif
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, pcl::PointXYZI * const po)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    #ifdef USE_IKFOM
    #ifdef Online_Calibration
    Eigen::Vector3d p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    #else
    Eigen::Vector3d p_global(state_point.rot * (p_body + Lid_offset_to_IMU) + state_point.pos);
    #endif
    #else
    #ifdef USE_QUA
    Eigen::Matrix3d rot_qua;
    rot_qua<<ROT_FROM_QUA(state.qua_end);
    #ifdef ON_CAL_QUA
    Eigen::Matrix3d rot_off;
    rot_off<<ROT_FROM_QUA(state.off_r);
    Eigen::Vector3d p_global(rot_qua * (rot_off*p_body + state.off_t) + state.pos_end);
    #else
    Eigen::Vector3d p_global(rot_qua * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif
    #else
    Eigen::Vector3d p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif
    #endif

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - std::floor(intensity);

    int reflection_map = intensity*10000;

    // //std::cout<<"DEBUG reflection_map "<<reflection_map<<std::endl;

    // if (reflection_map < 30)
    // {
    //     int green = (reflection_map * 255 / 30);
    //     po->r = 0;
    //     po->g = green & 0xff;
    //     po->b = 0xff;
    // }
    // else if (reflection_map < 90)
    // {
    //     int blue = (((90 - reflection_map) * 255) / 60);
    //     po->r = 0x0;
    //     po->g = 0xff;
    //     po->b = blue & 0xff;
    // }
    // else if (reflection_map < 150)
    // {
    //     int red = ((reflection_map-90) * 255 / 60);
    //     po->r = red & 0xff;
    //     po->g = 0xff;
    //     po->b = 0x0;
    // }
    // else
    // {
    //     int green = (((255-reflection_map) * 255) / (255-150));
    //     po->r = 0xff;
    //     po->g = green & 0xff;
    //     po->b = 0;
    // }
}

int cube_ind(const int &i, const int &j, const int &k)
{
    return (i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k);
}

bool CenterinFOV(Eigen::Vector3f cube_p)
{                                
    #ifdef USE_IKFOM
    Eigen::Vector3f dis_vec = (pos_lid).cast<float>() - cube_p;
    #else
    #ifdef ON_CAL_QUA
    Eigen::Matrix3d rot_off;
    rot_off<<ROT_FROM_QUA(state.qua_end);
    Eigen::Vector3d pos_LiD = state.pos_end + rot_off * state.off_t;
    Eigen::Vector3f dis_vec = pos_LiD.cast<float>() - cube_p;
    #else
    Eigen::Vector3f dis_vec = state.pos_end.cast<float>() - cube_p;
    #endif
    #endif
    float squaredSide1 = dis_vec.transpose() * dis_vec;

    if(squaredSide1 < 0.4 * cube_len * cube_len) return true;

    dis_vec = XAxisPoint_world.cast<float>() - cube_p;
    float squaredSide2 = dis_vec.transpose() * dis_vec;

    float ang_cos = fabs(squaredSide1 <= 3) ? 1.0 :
        (LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2) / (2 * LIDAR_SP_LEN * sqrt(squaredSide1));
    
    return ((ang_cos > HALF_FOV_COS)? true : false);
}

bool CornerinFOV(Eigen::Vector3f cube_p)
{                       
    #ifdef USE_IKFOM
    Eigen::Vector3f dis_vec = (pos_lid).cast<float>() - cube_p;
    #else     
    #ifdef ON_CAL_QUA
    Eigen::Matrix3d rot_off;
    rot_off<<ROT_FROM_QUA(state.qua_end);
    Eigen::Vector3d pos_LiD = state.pos_end + rot_off * state.off_t;
    Eigen::Vector3f dis_vec = pos_LiD.cast<float>() - cube_p;
    #else    
    Eigen::Vector3f dis_vec = state.pos_end.cast<float>() - cube_p;
    #endif
    #endif
    float squaredSide1 = dis_vec.transpose() * dis_vec;

    dis_vec = XAxisPoint_world.cast<float>() - cube_p;
    float squaredSide2 = dis_vec.transpose() * dis_vec;

    float ang_cos = fabs(squaredSide1 <= 3) ? 1.0 :
        (LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2) / (2 * LIDAR_SP_LEN * sqrt(squaredSide1));
    
    return ((ang_cos > HALF_FOV_COS)? true : false);
}

void lasermap_fov_segment()
{
    laserCloudValidNum = 0;

    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);

    #ifdef USE_IKFOM
    Eigen::Vector3d pos_LiD = pos_lid;
    #else
    #ifdef ON_CAL_QUA
    Eigen::Matrix3d rot_off;
    rot_off<<ROT_FROM_QUA(state.qua_end);
    Eigen::Vector3d pos_LiD = state.pos_end + rot_off * state.off_t;
    #else
    Eigen::Vector3d pos_LiD = state.pos_end;
    #endif
    #endif

    int centerCubeI = int((pos_LiD(0) + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
    int centerCubeJ = int((pos_LiD(1) + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
    int centerCubeK = int((pos_LiD(2) + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

    if (pos_LiD(0) + 0.5 * cube_len < 0) centerCubeI--;
    if (pos_LiD(1) + 0.5 * cube_len < 0) centerCubeJ--;
    if (pos_LiD(2) + 0.5 * cube_len < 0) centerCubeK--;

    bool last_inFOV_flag = 0;
    int  cube_index = 0;

    T2.push_back(Measures.lidar_beg_time);
    double t_begin = omp_get_wtime();

    std::cout<<"centerCubeIJK: "<<centerCubeI<<" "<<centerCubeJ<<" "<<centerCubeK<<std::endl;

    while (centerCubeI < FOV_RANGE + 1)
    {
        for (int j = 0; j < laserCloudHeight; j++)
        {
            for (int k = 0; k < laserCloudDepth; k++)
            {
                int i = laserCloudWidth - 1;

                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray[cube_ind(i, j, k)]  = featsArray[cube_ind(i-1, j, k)];
                    _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i-1, j, k)];
                }

                featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeI++;
        laserCloudCenWidth++;
    }

    while (centerCubeI >= laserCloudWidth - (FOV_RANGE + 1)) {
        for (int j = 0; j < laserCloudHeight; j++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int i = 0;

                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray[cube_ind(i, j, k)]  = featsArray[cube_ind(i+1, j, k)];
                    _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i+1, j, k)];
                }

                featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeI--;
        laserCloudCenWidth--;
    }

    while (centerCubeJ < (FOV_RANGE + 1)) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int j = laserCloudHeight - 1;

                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray[cube_ind(i, j, k)]  = featsArray[cube_ind(i, j-1, k)];
                    _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i, j-1, k)];
                }

                featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
    }

    while (centerCubeJ >= laserCloudHeight - (FOV_RANGE + 1)) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int j = 0;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray[cube_ind(i, j, k)]  = featsArray[cube_ind(i, j+1, k)];
                    _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i, j+1, k)];
                }

                featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
    }

    while (centerCubeK < (FOV_RANGE + 1)) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int j = 0; j < laserCloudHeight; j++) {
                int k = laserCloudDepth - 1;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray[cube_ind(i, j, k)]  = featsArray[cube_ind(i, j, k-1)];
                    _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i, j, k-1)];
                }

                featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK++;
        laserCloudCenDepth++;
    }

    while (centerCubeK >= laserCloudDepth - (FOV_RANGE + 1))
    {
        for (int i = 0; i < laserCloudWidth; i++)
        {
            for (int j = 0; j < laserCloudHeight; j++)
            {
                int k = 0;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray[cube_ind(i, j, k)]  = featsArray[cube_ind(i, j, k+1)];
                    _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i, j, k+1)];
                }

                featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK--;
        laserCloudCenDepth--;
    }

    cube_points_add->clear();
    featsFromMap->clear();
    bool now_inFOV[laserCloudNum] = {false};

    // std::cout<<"centerCubeIJK: "<<centerCubeI<<" "<<centerCubeJ<<" "<<centerCubeK<<std::endl;
    // std::cout<<"laserCloudCen: "<<laserCloudCenWidth<<" "<<laserCloudCenHeight<<" "<<laserCloudCenDepth<<std::endl;

    for (int i = centerCubeI - FOV_RANGE; i <= centerCubeI + FOV_RANGE; i++) 
    {
        for (int j = centerCubeJ - FOV_RANGE; j <= centerCubeJ + FOV_RANGE; j++) 
        {
            for (int k = centerCubeK - FOV_RANGE; k <= centerCubeK + FOV_RANGE; k++) 
            {
                if (i >= 0 && i < laserCloudWidth &&
                        j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth) 
                {
                    Eigen::Vector3f center_p(cube_len * (i - laserCloudCenWidth), \
                                             cube_len * (j - laserCloudCenHeight), \
                                             cube_len * (k - laserCloudCenDepth));

                    float check1, check2;
                    float squaredSide1, squaredSide2;
                    float ang_cos = 1;
                    bool &last_inFOV = _last_inFOV[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    bool inFOV = CenterinFOV(center_p);

                    for (int ii = -1; (ii <= 1) && (!inFOV); ii += 2) 
                    {
                        for (int jj = -1; (jj <= 1) && (!inFOV); jj += 2) 
                        {
                            for (int kk = -1; (kk <= 1) && (!inFOV); kk += 2) 
                            {
                                Eigen::Vector3f corner_p(cube_len * ii, cube_len * jj, cube_len * kk);
                                corner_p = center_p + 0.5 * corner_p;
                                
                                inFOV = CornerinFOV(corner_p);
                            }
                        }
                    }

                    now_inFOV[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = inFOV;

                #ifdef USE_ikdtree
                    /*** readd cubes and points ***/
                    if (inFOV)
                    {
                        int center_index = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        *cube_points_add += *featsArray[center_index];
                        featsArray[center_index]->clear();
                        if (!last_inFOV)
                        {
                            BoxPointType cub_points;
                            for(int i = 0; i < 3; i++)
                            {
                                cub_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
                                cub_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
                            }
                            cub_needad.push_back(cub_points);
                            laserCloudValidInd[laserCloudValidNum] = center_index;
                            laserCloudValidNum ++;
                            // std::cout<<"readd center: "<<center_p.transpose()<<std::endl;
                        }
                    }

                #else
                    if (inFOV)
                    {
                        int center_index = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        *featsFromMap += *featsArray[center_index];
                        laserCloudValidInd[laserCloudValidNum] = center_index;
                        laserCloudValidNum++;
                    }
                    last_inFOV = inFOV;
                #endif
                }
            }
        }
    }

    #ifdef USE_ikdtree
    cub_needrm.clear();
    cub_needad.clear();
    /*** delete cubes ***/
    for (int i = 0; i < laserCloudWidth; i++) 
    {
        for (int j = 0; j < laserCloudHeight; j++) 
        {
            for (int k = 0; k < laserCloudDepth; k++) 
            {
                int ind = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                if((!now_inFOV[ind]) && _last_inFOV[ind])
                {
                    BoxPointType cub_points;
                    Eigen::Vector3f center_p(cube_len * (i - laserCloudCenWidth),\
                                             cube_len * (j - laserCloudCenHeight),\
                                             cube_len * (k - laserCloudCenDepth));
                    // std::cout<<"center_p: "<<center_p.transpose()<<std::endl;

                    for(int i = 0; i < 3; i++)
                    {
                        cub_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
                        cub_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
                    }
                    cub_needrm.push_back(cub_points);
                }
                _last_inFOV[ind] = now_inFOV[ind];
            }
        }
    }
    #endif

    copy_time = omp_get_wtime() - t_begin;

#ifdef USE_ikdtree
    if(cub_needrm.size() > 0)               ikdtree.Delete_Point_Boxes(cub_needrm);
    // s_plot4.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if(cub_needad.size() > 0)               ikdtree.Add_Point_Boxes(cub_needad); 
    // s_plot5.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if(cube_points_add->points.size() > 0)  ikdtree.Add_Points(cube_points_add->points, true);
#endif
    // s_plot6.push_back(omp_get_wtime() - t_begin);
    // readd_time = omp_get_wtime() - t_begin - copy_time;
}

void feat_points_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    // std::cout<<"got feature"<<std::endl;
    if (msg->header.stamp.toSec() < last_timestamp_lidar) // - 1621800000 ; - 1621436000  1622303210(e16) - 1621436000   - 1622378400(0530e2)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
    lidar_buffer.push_back(msg);
    last_timestamp_lidar = msg->header.stamp.toSec(); // - 1622378400;// - 1621436000; // - 1622179354; // - 1621436000; // - 1621800000;  - 1621436000

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1622179354; // - 1621436000; // - 1621800000;

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    // std::cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<std::endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

#ifdef USE_IKFOM
#ifdef RESTORE_VICON
void vicon_pose_cbk(const geometry_msgs::PoseStamped::ConstPtr &msg_in) 
{
    geometry_msgs::PoseStamped::Ptr msg(new geometry_msgs::PoseStamped(*msg_in));

    double timestamp = msg->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1621436000; //- 1621800000;

    mtx_buffer.lock();

    if (timestamp < last_timestamp_vicon_pose)
    {
        ROS_ERROR("vicon loop back, clear");
        // imu_buffer.clear();
        // flg_reset = true;
    }

    last_timestamp_vicon_pose = timestamp;
    v_time.push_back(timestamp);
    v_pose(0) = msg_in->pose.position.x;
    v_pose(1) = msg_in->pose.position.y;
    v_pose(2) = msg_in->pose.position.z;
    // v_rot.w = msg_in->pose.orientation.w;
    // v_rot.x = msg_in->pose.orientation.x;
    // v_rot.y = msg_in->pose.orientation.y;
    // v_rot.z = msg_in->pose.orientation.z;
    if(!vicon_first)
    {
        init_pose = v_pose;
        // init_rot = SO3(msg_in->pose.orientation.w, msg_in->pose.orientation.x, msg_in->pose.orientation.y, msg_in->pose.orientation.z);
        Eigen::Matrix3d init_rotm;
        init_rotm<<0,1,0,-1,0,0,0,0,1;
        init_rot = SO3(init_rotm);
        vicon_first = 1;
    }
    // v_pose = init_rot.conjugate() * (v_pose - init_pose);
    v_pose = init_rot * (v_pose - init_pose);
    // v_rot = SO3ToEuler(init_rot.conjugate() * SO3(msg_in->pose.orientation.w, msg_in->pose.orientation.x, msg_in->pose.orientation.y, msg_in->pose.orientation.z));
    v_rot = SO3ToEuler(SO3(msg_in->pose.orientation.w, msg_in->pose.orientation.x, msg_in->pose.orientation.y, msg_in->pose.orientation.z));
    pose_g.push_back(v_pose);
    rot_g.push_back(init_rot * v_rot);
    // imu_buffer.push_back(msg);
    // std::cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<std::endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void vicon_vel_cbk(const geometry_msgs::TwistStamped::ConstPtr &msg_in) 
{
    if(!vicon_first) return;
    geometry_msgs::TwistStamped::Ptr msg(new geometry_msgs::TwistStamped(*msg_in));

    double timestamp = msg->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1621436000; // - 1621800000;

    mtx_buffer.lock();

    if (timestamp < last_timestamp_vicon_vel)
    {
        ROS_ERROR("vicon loop back, clear");
        // imu_buffer.clear();
        // flg_reset = true;
    }

    last_timestamp_vicon_vel = timestamp;
    // v_time.push_back(timestamp);
    v_vel(0) = msg_in->twist.linear.x;
    v_vel(1) = msg_in->twist.linear.y;
    v_vel(2) = msg_in->twist.linear.z;
    // v_vel = init_rot.conjugate() * v_vel;
    v_vel = init_rot * v_vel;
    if(v_vel[2]>50)
    {
        v_vel[0] = v_vel[1] = v_vel[2] = 0.0;
    }
    vel_g.push_back(v_vel);
    // imu_buffer.push_back(msg);
    // std::cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<std::endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
#endif
#endif

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
        meas.lidar_beg_time = lidar_buffer.front()->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1622179354; // - 1621436000; // - 1621800000;
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1622179354; // - 1621436000; // - 1621800000;
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1622179354; // - 1621800000;
        if(imu_time > lidar_end_time + 0.02) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    lidar_pushed = false;
    // if (meas.imu.empty()) return false;
    // std::cout<<"[IMU Sycned]: "<<imu_time<<" "<<lidar_end_time<<std::endl;
    return true;
}

#ifdef USE_IKFOM
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    coeffSel->clear(); 
    total_residual = 0.0; 
    /** closest surface search and residual computation **/
    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down->points[i]; 
        PointType &pointSel_tmpt = feats_down_updated->points[i]; 
        //double search_start = omp_get_wtime();
        /* transform to world frame */
        Eigen::Vector3d p_body(point_body.x, point_body.y, point_body.z);
        #ifdef Online_Calibration
        Eigen::Vector3d p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        #else
        Eigen::Vector3d p_global(s.rot * (p_body + Lid_offset_to_IMU) + s.pos);
        #endif
        pointSel_tmpt.x = p_global(0);
        pointSel_tmpt.y = p_global(1);
        pointSel_tmpt.z = p_global(2);
        pointSel_tmpt.intensity = point_body.intensity;

        std::vector<float> pointSearchSqDis_surf(NUM_MATCH_POINTS);
        #ifdef USE_ikdtree
        auto &points_near = Nearest_Points[i];
        #else
        auto &points_near = pointSearchInd_surf[i];
        #endif
        if (ekfom_data.converge)
        {
            point_selected_surf[i] = true;
        /** Find the closest surfaces in the map **/
        #ifdef USE_ikdtree
            ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
        #else
            kdtreeSurfFromMap->nearestKSearch(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
        #endif
            float max_distance = pointSearchSqDis_surf[NUM_MATCH_POINTS - 1];
            if (max_distance >= 6.0) // > 3
            {
                point_selected_surf[i] = false;
            }
        }

        if (!point_selected_surf[i]) continue;

        double pca_start = omp_get_wtime();

        /// PCA (using minimum square method)
        cv::Mat matA0(NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(-1));
        cv::Mat matX0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(0));
                        
        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
        #ifdef USE_ikdtree
            matA0.at<float>(j, 0) = points_near[j].x;
            matA0.at<float>(j, 1) = points_near[j].y;
            matA0.at<float>(j, 2) = points_near[j].z;
        #else
            matA0.at<float>(j, 0) = featsFromMap->points[points_near[j]].x;
            matA0.at<float>(j, 1) = featsFromMap->points[points_near[j]].y;
            matA0.at<float>(j, 2) = featsFromMap->points[points_near[j]].z;
        #endif
        }

        cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);  //TODO

        float pa = matX0.at<float>(0, 0);
        float pb = matX0.at<float>(1, 0);
        float pc = matX0.at<float>(2, 0);
        float pd = 1;

        //ps is the norm of the plane norm_vec vector
        //pd is the distance from point to plane
        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
        #ifdef USE_ikdtree
            if (fabs(pa * points_near[j].x +
                        pb * points_near[j].y +
                        pc * points_near[j].z + pd) > 0.1)
        #else
            if (fabs(pa * featsFromMap->points[points_near[j]].x +
                        pb * featsFromMap->points[points_near[j]].y +
                        pc * featsFromMap->points[points_near[j]].z + pd) > 0.1)
        #endif
            {
                planeValid = false;
                point_selected_surf[i] = false;
                break;
            }
        }

        if (planeValid) 
        {
            //loss fuction
            float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
            //if(fabs(pd2) > 0.1) continue;
            float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y + pointSel_tmpt.z * pointSel_tmpt.z));

            if ((s > 0.92))// && ((std::abs(pd2) - res_last[i]) < 3 * res_mean_last)) > 0.85
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
                
                // if(i%50==0) std::cout<<"s: "<<s<<"last res: "<<res_last[i]<<" current res: "<<std::abs(pd2)<<std::endl;
                res_last[i] = std::abs(pd2);
            }
            else
            {
                point_selected_surf[i] = false;
            }
        }
        // pca_time += omp_get_wtime() - pca_start;
    }

    laserCloudSelNum = 0;

    for (int i = 0; i < coeffSel_tmpt->points.size(); i++)
    {
        float error_abs = std::abs(coeffSel_tmpt->points[i].intensity);               
        if (point_selected_surf[i]) // && (error_abs < 0.5)) // res_last[i] <= 2.0
        {
            laserCloudOri->push_back(feats_down->points[i]);
            coeffSel->push_back(coeffSel_tmpt->points[i]);
            total_residual += res_last[i];
            laserCloudSelNum ++;
        }
    }

    res_mean_last = total_residual / laserCloudSelNum;
    std::cout << "[ mapping ]: Effective feature num: "<<laserCloudSelNum<<" res_mean_last "<<res_mean_last<<std::endl;

    //match_time += omp_get_wtime() - match_start;
    
    //double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    //MatrixXd H(effct_feat_num, 23);
    #ifdef Online_Calibration
    ekfom_data.h_x = Eigen::MatrixXd::Zero(laserCloudSelNum, 12); //23
    #else
    ekfom_data.h_x = Eigen::MatrixXd::Zero(laserCloudSelNum, 6); //23
    #endif
    ekfom_data.h.resize(laserCloudSelNum); // = VectorXd::Zero(effct_feat_num);
    //VectorXd meas_vec(effct_feat_num);

    for (int i = 0; i < laserCloudSelNum; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        Eigen::Vector3d point_this_be(laser_p.x, laser_p.y, laser_p.z);
        #ifdef Online_Calibration
        Eigen::Matrix3d point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        Eigen::Vector3d point_this = s.offset_R_L_I * point_this_be +s.offset_T_L_I;
        #else
        Eigen::Vector3d point_this = point_this_be + Lid_offset_to_IMU;
        #endif
        Eigen::Matrix3d point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = coeffSel->points[i];
        Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        Eigen::Vector3d C(s.rot.conjugate() *norm_vec);
        Eigen::Vector3d A(point_crossmat * C); // s.rot.conjugate() * norm_vec);
        #ifdef Online_Calibration
        Eigen::Vector3d B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
        ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        #else
        ekfom_data.h_x.block<1, 6>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A); //, VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        #endif
        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
}
#endif    

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    ros::Subscriber sub_pcl = nh.subscribe("/laser_cloud_flat", 20000, feat_points_cbk);
    ros::Subscriber sub_imu = nh.subscribe("/livox/imu", 20000, imu_cbk);
    #ifdef USE_IKFOM
    #ifdef RESTORE_VICON
    ros::Subscriber sub_vicon_pose = nh.subscribe("/mavros/vision_pose/pose", 20000, vicon_pose_cbk);
    ros::Subscriber sub_vicon_vel = nh.subscribe("/mavros/vision_vel/vel", 20000, vicon_vel_cbk);
    #endif
    #endif
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

    /*** variables definition ***/
    bool dense_map_en, flg_EKF_inited = 0, flg_map_inited = 0, flg_EKF_converged = 0;
    std::string map_file_path;
    int effect_feat_num = 0, frame_num = 0;
    double filter_size_corner_min, filter_size_surf_min, filter_size_map_min, fov_deg,\
           deltaT, deltaR, aver_time_consu = 0, first_lidar_time = 0, total_time_consu = 0; // aver_time_consu_predict = 0;
    #ifndef USE_IKFOM
    Eigen::Matrix<double,DIM_OF_STATES,DIM_OF_STATES> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
    #endif

    nav_msgs::Odometry odomAftMapped;

    cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

    PointType pointOri, pointSel, coeff;
    PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
    feats_down->clear();
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

    HALF_FOV_COS = std::cos((fov_deg + 10.0) * 0.5 * PI_M / 180.0);

    for (int i = 0; i < laserCloudNum; i++)
    {
        featsArray[i].reset(new PointCloudXYZI());
    }

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    #ifdef USE_IKFOM
    #ifdef Online_Calibration
    double epsi[23] = {0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001};
    #else
    double epsi[17] = {0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
    #endif
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
    state_ikfom init_state = kf.get_x();
    #ifdef Online_Calibration
    init_state.offset_T_L_I = Lid_offset_to_IMU;
    init_state.offset_R_L_I = Lid_rot_to_IMU;
    #endif
    kf.change_x(init_state);
    // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf.get_P() * 0.001;
    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf.get_P() * 0.0001;
    #ifdef Online_Calibration
    // init_P(0,0)=init_P(1,1)=init_P(2,2)=init_P(3,3)=init_P(4,4)=init_P(5,5)=init_P(12,12)=init_P(13,13)=init_P(14,14)=1.0;
    // init_P(6,6)=init_P(7,7)=init_P(8,8)=0.001;
    // init_P(9,9)=init_P(10,10)=init_P(11,11)=0.01;
    init_P(15,15)=init_P(16,16)=init_P(17,17)=0.001;
    init_P(18,18)=init_P(19,19)=init_P(20,20)=0.03;
    #else
    init_P(0,0)=init_P(1,1)=init_P(2,2)=init_P(3,3)=init_P(4,4)=init_P(5,5)=init_P(6,6)=init_P(7,7)=init_P(8,8)=1.0;
    #endif
    kf.change_P(init_P);
    #endif

    std::shared_ptr<ImuProcess> p_imu(new ImuProcess());

    /*** debug record ***/
    std::ofstream fout_pre, fout_out;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),std::ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),std::ios::out);
    if (fout_pre && fout_out)
        std::cout << "~~~~"<<ROOT_DIR<<" file opened" << std::endl;
    else
        std::cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << std::endl;

//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        while(sync_packages(Measures)) 
        {
            if (flg_reset)
            {
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, match_time, solve_start, solve_time, pca_time, svd_time;
            match_time = 0;
            solve_time = 0;
            pca_time   = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();
            #ifdef USE_IKFOM
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            #ifdef Online_Calibration
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            #else
            pos_lid = state_point.pos + state_point.rot * Lid_offset_to_IMU;
            #endif
            #else
            p_imu->Process(Measures, state, feats_undistort);
            StatesGroup state_propagat(state);
            #endif

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                first_lidar_time = Measures.lidar_beg_time;
                std::cout<<"not ready for odometry"<<std::endl;
                continue;
            }

            if ((Measures.lidar_beg_time - first_lidar_time) < INIT_TIME)
            {
                flg_EKF_inited = false;
                std::cout<<"||||||||||Initiallizing LiDar||||||||||"<<std::endl;
            }
            else
            {
                flg_EKF_inited = true;
            }
            
            /*** Compute the euler angle ***/
        #ifdef DEBUG_PRINT
            #ifdef USE_IKFOM
            euler_cur = SO3ToEuler(state_point.rot);
            cout<<"current lidar time "<<Measures.lidar_beg_time<<" "<<"first lidar time "<<first_lidar_time<<endl;
            cout<<"pre-integrated states: "<<euler_cur.transpose()*57.3<<" "<<state_point.pos.transpose()<<" "<<state_point.vel.transpose()<<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<endl;
            #else
            #ifdef USE_QUA
            Eigen::Matrix3d rot_qua;
            rot_qua<<ROT_FROM_QUA(state.qua_end);
            euler_cur = RotMtoEuler(rot_qua);
            #else
            euler_cur = RotMtoEuler(state.rot_end);
            #endif
            cout<<"current lidar time "<<Measures.lidar_beg_time<<" "<<"first lidar time "<<first_lidar_time<<endl;
            cout<<"pre-integrated states: "<<euler_cur.transpose()*57.3<<" "<<state.pos_end.transpose()<<" "<<state.vel_end.transpose()<<" "<<state.bias_g.transpose()<<" "<<state.bias_a.transpose()<<endl;
            #endif
        #endif
            
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();
            
            /*** downsample the features of new frame ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down);

        #ifdef USE_ikdtree
            /*** initialize the map kdtree ***/
            if((feats_down->points.size() > 1) && (ikdtree.Root_Node == nullptr))
            {
                // std::vector<PointType> points_init = feats_down->points;
                ikdtree.set_downsample_param(filter_size_map_min);
                ikdtree.Build(feats_down->points);
                flg_map_inited = true;
                continue;
            }

            if(ikdtree.Root_Node == nullptr)
            {
                flg_map_inited = false;
                std::cout<<"~~~~~~~Initiallize Map iKD-Tree Failed!"<<std::endl;
                continue;
            }
            int featsFromMapNum = ikdtree.size();
        #else
            if(featsFromMap->points.empty())
            {
                downSizeFilterMap.setInputCloud(feats_down);
            }
            else
            {
                downSizeFilterMap.setInputCloud(featsFromMap);
            }
            downSizeFilterMap.filter(*featsFromMap);
            int featsFromMapNum = featsFromMap->points.size();
        #endif
            feats_down_size = feats_down->points.size();
            std::cout<<"[ mapping ]: Raw feature num: "<<feats_undistort->points.size()<<" downsamp num "<<feats_down_size<<" Map num: "<<featsFromMapNum<<" laserCloudValidNum "<<laserCloudValidNum<<std::endl;

            /*** ICP and iterated Kalman filter update ***/
            coeffSel_tmpt->resize(feats_down_size);
            feats_down_updated->resize(feats_down_size);
            res_last.resize(feats_down_size, 1000.0); // initial

            if (featsFromMapNum >= 5)
            {
                t1 = omp_get_wtime();      
                      
            #ifdef USE_ikdtree
                //std::vector<PointVector> Nearest_Points(feats_down_size);
            #else
                kdtreeSurfFromMap->setInputCloud(featsFromMap);
                
            #endif

                point_selected_surf.resize(feats_down_size, true);
                pointSearchInd_surf.resize(feats_down_size);
                
                int  rematch_num = 0;
                bool rematch_en = 0;
                flg_EKF_converged = 0;
                deltaR = 0.0;
                deltaT = 0.0;
                t2 = omp_get_wtime();
            #ifdef USE_IKFOM
                // double solve_H_time = 0;
                #ifdef Online_Calibration
                kf.update_iterated_dyn_share_modified_extrinsic(LASER_POINT_COV); //, solve_H_time);
                #else
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV); //, solve_H_time);
                #endif
                //state_ikfom updated_state = kf.get_x();
                state_point = kf.get_x();
                //euler_cur = RotMtoEuler(state_point.rot.toRotationMatrix());
                euler_cur = SO3ToEuler(state_point.rot);
                #ifdef Online_Calibration
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                #else
                pos_lid = state_point.pos + state_point.rot * Lid_offset_to_IMU;
                #endif
                geoQuat.x = state_point.rot.coeffs()[0];
                geoQuat.y = state_point.rot.coeffs()[1];
                geoQuat.z = state_point.rot.coeffs()[2];
                geoQuat.w = state_point.rot.coeffs()[3];
                
            
            #else
                for (iterCount = -1; iterCount < NUM_MAX_ITERATIONS; iterCount++) 
                {
                    match_start = omp_get_wtime();
                    laserCloudOri->clear();
                    coeffSel->clear();

                    /** closest surface search and residual computation **/
                    omp_set_num_threads(4);
                    #pragma omp parallel for
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        PointType &pointOri_tmpt = feats_down->points[i];
                        PointType &pointSel_tmpt = feats_down_updated->points[i];

                        /* transform to world frame */
                        pointBodyToWorld(&pointOri_tmpt, &pointSel_tmpt);
                        std::vector<float> pointSearchSqDis_surf;
                    #ifdef USE_ikdtree
                        auto &points_near = Nearest_Points[i];
                    #else
                        auto &points_near = pointSearchInd_surf[i];
                    #endif
                        
                        if (iterCount == -1 || rematch_en)
                        {
                            point_selected_surf[i] = true;
                            /** Find the closest surfaces in the map **/
                        #ifdef USE_ikdtree
                            ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
                        #else
                            kdtreeSurfFromMap->nearestKSearch(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
                        #endif
                            float max_distance = pointSearchSqDis_surf[NUM_MATCH_POINTS - 1];
                        
                            if (max_distance >= 6.0) // >3
                            {
                                point_selected_surf[i] = false;
                            }
                        }

                        if (point_selected_surf[i] == false) continue;

                        // match_time += omp_get_wtime() - match_start;

                        double pca_start = omp_get_wtime();

                        /// PCA (using minimum square method)
                        cv::Mat matA0(NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all(0));
                        cv::Mat matB0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(-1));
                        cv::Mat matX0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(0));
                        
                        for (int j = 0; j < NUM_MATCH_POINTS; j++)
                        {
                        #ifdef USE_ikdtree
                            matA0.at<float>(j, 0) = points_near[j].x;
                            matA0.at<float>(j, 1) = points_near[j].y;
                            matA0.at<float>(j, 2) = points_near[j].z;
                        #else
                            matA0.at<float>(j, 0) = featsFromMap->points[points_near[j]].x;
                            matA0.at<float>(j, 1) = featsFromMap->points[points_near[j]].y;
                            matA0.at<float>(j, 2) = featsFromMap->points[points_near[j]].z;
                        #endif
                        }

                        //matA0*matX0=matB0
                        //AX+BY+CZ+D = 0 <=> AX+BY+CZ=-D <=> (A/D)X+(B/D)Y+(C/D)Z = -1
                        //(X,Y,Z)<=>mat_a0
                        //A/D, B/D, C/D <=> mat_x0
            
                        cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);  //TODO

                        float pa = matX0.at<float>(0, 0);
                        float pb = matX0.at<float>(1, 0);
                        float pc = matX0.at<float>(2, 0);
                        float pd = 1;

                        //ps is the norm of the plane norm_vec vector
                        //pd is the distance from point to plane
                        float ps = sqrt(pa * pa + pb * pb + pc * pc);
                        pa /= ps;
                        pb /= ps;
                        pc /= ps;
                        pd /= ps;

                        bool planeValid = true;
                        for (int j = 0; j < NUM_MATCH_POINTS; j++)
                        {
                        #ifdef USE_ikdtree
                            if (fabs(pa * points_near[j].x +
                                        pb * points_near[j].y +
                                        pc * points_near[j].z + pd) > 0.1)
                        #else
                            if (fabs(pa * featsFromMap->points[points_near[j]].x +
                                        pb * featsFromMap->points[points_near[j]].y +
                                        pc * featsFromMap->points[points_near[j]].z + pd) > 0.1)
                        #endif
                            {
                                planeValid = false;
                                point_selected_surf[i] = false;
                                break;
                            }
                        }

                        if (planeValid) 
                        {
                            //loss fuction
                            float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                            //if(fabs(pd2) > 0.1) continue;
                            float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y + pointSel_tmpt.z * pointSel_tmpt.z));

                            if ((s > 0.92))// && ((std::abs(pd2) - res_last[i]) < 3 * res_mean_last)) >0.85
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
                                
                                // if(i%50==0) std::cout<<"s: "<<s<<"last res: "<<res_last[i]<<" current res: "<<std::abs(pd2)<<std::endl;
                                res_last[i] = std::abs(pd2);
                            }
                            else
                            {
                                point_selected_surf[i] = false;
                            }
                        }

                        pca_time += omp_get_wtime() - pca_start;
                    }

                    total_residual = 0.0;
                    laserCloudSelNum = 0;

                    for (int i = 0; i < coeffSel_tmpt->points.size(); i++)
                    {
                        if (point_selected_surf[i]) // && (res_last[i] <= 2.0))
                        {
                            laserCloudOri->push_back(feats_down->points[i]);
                            coeffSel->push_back(coeffSel_tmpt->points[i]);
                            total_residual += res_last[i];
                            laserCloudSelNum ++;
                        }
                    }

                    res_mean_last = total_residual / laserCloudSelNum;
                    std::cout << "[ mapping ]: Effective feature num: "<<laserCloudSelNum<<" res_mean_last "<<res_mean_last<<std::endl;

                    match_time += omp_get_wtime() - match_start;
                    solve_start = omp_get_wtime();
                    
                    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
                    #ifdef USE_QUA
                    #ifdef ON_CAL_QUA
                    Eigen::VectorXd meas_vec(laserCloudSelNum + 2);
                    Eigen::MatrixXd Hsub(laserCloudSelNum + 2, 26);
                    Eigen::Matrix3d rot_off, rot_qua;
                    rot_off<<ROT_FROM_QUA(state.off_r); 
                    rot_qua<<ROT_FROM_QUA(state.qua_end);
                    #else
                    Eigen::VectorXd meas_vec(laserCloudSelNum + 1);
                    Eigen::MatrixXd Hsub(laserCloudSelNum + 1, 7);
                    #endif
                    #else
                    Eigen::MatrixXd Hsub(laserCloudSelNum, 6);
                    Eigen::VectorXd meas_vec(laserCloudSelNum);
                    #endif
                    // Eigen::VectorXd meas_vec(laserCloudSelNum);
                    // Hsub.setZero();
                    #ifdef USE_QUA
                    #ifdef ON_CAL_QUA
                    Hsub.row(laserCloudSelNum) << 2*state.qua_end[0], 2*state.qua_end[1], 2*state.qua_end[2], 2*state.qua_end[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                    Hsub.row(laserCloudSelNum + 1) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*state.off_r[0], 2*state.off_r[1], 2*state.off_r[2], 2*state.off_r[3], 0, 0, 0;
                    meas_vec(laserCloudSelNum) = 1 - state.qua_end.norm() * state.qua_end.norm();
                    meas_vec(laserCloudSelNum + 1) = 1 - state.off_r.norm() * state.off_r.norm();
                    #else
                    Hsub.row(laserCloudSelNum) << 2*state.qua_end[0], 2*state.qua_end[1], 2*state.qua_end[2], 2*state.qua_end[3], 0, 0, 0;
                    meas_vec(laserCloudSelNum) = 1 - state.qua_end.norm() * state.qua_end.norm();
                    #endif
                    #endif
                    // omp_set_num_threads(4);
                    // #pragma omp parallel for
                    for (int i = 0; i < laserCloudSelNum; i++)
                    {
                        const PointType &laser_p  = laserCloudOri->points[i];
                        Eigen::Vector3d point_this(laser_p.x, laser_p.y, laser_p.z);
                        #ifdef ON_CAL_QUA 
                        Eigen::Vector3d point_this_aft = rot_off * point_this + state.off_t;
                        #else
                        point_this += Lidar_offset_to_IMU;
                        #endif
                        #ifdef USE_QUA
                        #ifdef ON_CAL_QUA
                        Eigen::Matrix<double, 3, 4> z_qua = qua_vec_qua(state.qua_end, point_this_aft);
                        Eigen::Matrix<double, 3, 4> z_off = qua_vec_qua(state.off_r, point_this);
                        #else
                        Eigen::Matrix<double, 3, 4> z_qua = qua_vec_qua(state.qua_end, point_this);
                        #endif
                        #else
                        Eigen::Matrix3d point_crossmat;
                        point_crossmat<<SKEW_SYM_MATRX(point_this);
                        #endif
                        /*** get the normal vector of closest surface/corner ***/
                        const PointType &norm_p = coeffSel->points[i];
                        Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

                        /*** calculate the Measuremnt Jacobian matrix H ***/
                        #ifdef USE_QUA
                        #ifdef ON_CAL_QUA
                        Eigen::Vector4d A(z_qua.transpose()*norm_vec);
                        Eigen::Vector4d B(z_off.transpose()*rot_qua.transpose()*norm_vec);
                        Eigen::Vector3d C(rot_qua.transpose()*norm_vec);
                        Hsub.row(i) << VEC4_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, VEC4_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
                        #else
                        Eigen::Vector4d A(z_qua.transpose()*norm_vec);
                        Hsub.row(i) << VEC4_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;
                        #endif
                        #else
                        Eigen::Vector3d A(point_crossmat * state.rot_end.transpose() * norm_vec);
                        Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;
                        #endif

                        /*** Measuremnt: distance to the closest surface/corner ***/
                        meas_vec(i) = - norm_p.intensity;
                    }

                    #ifdef USE_QUA
                    Eigen::Vector4d rot_add;
                    Eigen::Vector3d t_add, v_add, bg_add, ba_add, g_add;
                    #else
                    Eigen::Vector3d rot_add, t_add, v_add, bg_add, ba_add, g_add;
                    #endif
                    Eigen::Matrix<double, DIM_OF_STATES, 1> solution;

                    #ifdef USE_QUA
                    #ifdef ON_CAL_QUA
                    Eigen::MatrixXd K(DIM_OF_STATES, laserCloudSelNum+2);
                    #else
                    Eigen::MatrixXd K(DIM_OF_STATES, laserCloudSelNum+1);
                    #endif
                    #else
                    Eigen::MatrixXd K(DIM_OF_STATES, laserCloudSelNum);
                    #endif
                    
                    /*** Iterative Kalman Filter Update ***/
                    if (!flg_EKF_inited)
                    {
                        #ifndef USE_QUA
                        /*** only run in initialization period ***/
                        Eigen::MatrixXd H_init(Eigen::Matrix<double, 9, DIM_OF_STATES>::Zero());
                        Eigen::MatrixXd z_init(Eigen::Matrix<double, 9, 1>::Zero());
                        H_init.block<3,3>(0,0)  = Eigen::Matrix3d::Identity();
                        H_init.block<3,3>(3,3)  = Eigen::Matrix3d::Identity();
                        H_init.block<3,3>(6,15) = Eigen::Matrix3d::Identity();
                        z_init.block<3,1>(0,0)  = - Log(state.rot_end);
                        z_init.block<3,1>(0,0)  = - state.pos_end;

                        auto H_init_T = H_init.transpose();
                        auto &&K_init = state.cov * H_init_T * (H_init * state.cov * H_init_T + 0.0001 * Eigen::Matrix<double,9,9>::Identity()).inverse();
                        solution      = K_init * z_init;

                        solution.block<9,1>(0,0).setZero();
                        state += solution;
                        state.cov = (Eigen::MatrixXd::Identity(DIM_OF_STATES, DIM_OF_STATES) - K_init * H_init) * state.cov;
                        #endif
                    }
                    else
                    {
                        auto &&Hsub_T = Hsub.transpose();
                        #ifdef USE_QUA
                        #ifdef ON_CAL_QUA
                        H_T_H = Hsub_T * Hsub;
                        #else
                        H_T_H.block<7,7>(0,0) = Hsub_T * Hsub;
                        #endif
                        #else
                        H_T_H.block<6,6>(0,0) = Hsub_T * Hsub;
                        #endif
                        Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> &&K_1 = \
                                    (H_T_H + (state.cov / LASER_POINT_COV).inverse()).inverse();
                        #ifdef USE_QUA
                        #ifdef ON_CAL_QUA
                        K = K_1 * Hsub_T;
                        #else
                        K = K_1.block<DIM_OF_STATES,7>(0,0) * Hsub_T;
                        #endif
                        #else
                        K = K_1.block<DIM_OF_STATES,6>(0,0) * Hsub_T;
                        #endif
                        // solution = K * meas_vec;
                        // state += solution;

                        auto vec = state_propagat - state;
                        // solution = K * (meas_vec - Hsub * vec.block<6,1>(0,0));
                        // state = state_propagat + solution;
                        #ifdef USE_QUA
                        #ifdef ON_CAL_QUA
                        solution = K * meas_vec + vec - K * Hsub * vec;
                        #else
                        solution = K * meas_vec + vec - K * Hsub * vec.block<7,1>(0,0);
                        #endif
                        #else
                        solution = K * meas_vec + vec - K * Hsub * vec.block<6,1>(0,0);
                        #endif
                        // std::cout << state.qua_end[0] << ";" << state.qua_end[1] << ";" << state.qua_end[2] << ";" << state.qua_end[3] << ";" << std::endl;
                        state += solution;
                        // std::cout << state.qua_end[0] << ";" << state.qua_end[1] << ";" << state.qua_end[2] << ";" << state.qua_end[3] << ";" << std::endl;
                        #ifdef USE_QUA
                        rot_add = solution.block<4,1>(0,0);
                        t_add   = solution.block<3,1>(4,0);
                        #else
                        rot_add = solution.block<3,1>(0,0);
                        t_add   = solution.block<3,1>(3,0);
                        #endif

                        flg_EKF_converged = false;

                        if ((rot_add.norm() * 57.3 < 0.01) \
                           && (t_add.norm() * 100 < 0.015))
                        {
                            flg_EKF_converged = true;
                        }  // 0.01 and 0.015

                        deltaR = rot_add.norm() * 57.3;
                        deltaT = t_add.norm() * 100;
                    }
                    #ifdef USE_QUA
                    Eigen::Matrix3d rot_qua_aft;
                    rot_qua_aft<<ROT_FROM_QUA(state.qua_end);
                    euler_cur = RotMtoEuler(rot_qua_aft);
                    #else
                    euler_cur = RotMtoEuler(state.rot_end);
                    #endif
                    #ifdef DEBUG_PRINT
                    std::cout<<"update: R"<<euler_cur.transpose()*57.3<<" p "<<state.pos_end.transpose()<<" v "<<state.vel_end.transpose()<<" bg"<<state.bias_g.transpose()<<" ba"<<state.bias_a.transpose()<<std::endl;
                    std::cout<<"dR & dT: "<<deltaR<<" "<<deltaT<<" res norm:"<<res_mean_last<<std::endl;
                    #endif

                    /*** Rematch Judgement ***/
                    rematch_en = false;
                    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2))))
                    {
                        rematch_en = true;
                        rematch_num ++;
                        std::cout<<"rematch_num: "<<rematch_num<<std::endl;
                    }

                    /*** Convergence Judgements and Covariance Update ***/
                    if (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))
                    {
                        if (flg_EKF_inited)
                        {
                            #ifdef USE_QUA
                            /*** Covariance Update ***/
                            #ifdef ON_CAL_QUA
                            G = K * Hsub;
                            #else
                            G.block<DIM_OF_STATES,7>(0,0) = K * Hsub;
                            #endif
                            state.cov = (I_STATE - G) * state.cov;
                            #else
                            /*** Covariance Update ***/
                            G.block<DIM_OF_STATES,6>(0,0) = K * Hsub;
                            state.cov = (I_STATE - G) * state.cov;
                            #endif
                            total_distance += (state.pos_end - position_last).norm();
                            position_last = state.pos_end;
                            geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));

                            // std::cout<<"position: "<<state.pos_end.transpose()<<" total distance: "<<total_distance<<std::endl;
                        }
                        solve_time += omp_get_wtime() - solve_start;
                        break;
                    }
                    solve_time += omp_get_wtime() - solve_start;
                }
            #endif
                
                std::cout<<"[ mapping ]: iteration count: "<<iterCount+1<<std::endl;

                t3 = omp_get_wtime();

                /*** add new frame points to map ikdtree ***/
            #ifdef USE_ikdtree
                PointVector points_history;
                ikdtree.acquire_removed_points(points_history);
                
                memset(cube_updated, 0, sizeof(cube_updated));
                
                for (int i = 0; i < points_history.size(); i++)
                {
                    PointType &pointSel = points_history[i];

                    int cubeI = int((pointSel.x + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
                    int cubeJ = int((pointSel.y + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
                    int cubeK = int((pointSel.z + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

                    if (pointSel.x + 0.5 * cube_len < 0) cubeI--;
                    if (pointSel.y + 0.5 * cube_len < 0) cubeJ--;
                    if (pointSel.z + 0.5 * cube_len < 0) cubeK--;

                    if (cubeI >= 0 && cubeI < laserCloudWidth &&
                            cubeJ >= 0 && cubeJ < laserCloudHeight &&
                            cubeK >= 0 && cubeK < laserCloudDepth) 
                    {
                        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                        featsArray[cubeInd]->push_back(pointSel);
                        
                    }
                }

                // omp_set_num_threads(4);
                // #pragma omp parallel for
                for (int i = 0; i < feats_down_size; i++)
                {
                    /* transform to world frame */
                    pointBodyToWorld(&(feats_down->points[i]), &(feats_down_updated->points[i]));
                }
                t4 = omp_get_wtime();
                ikdtree.Add_Points(feats_down_updated->points, true);
            #else
                bool cube_updated[laserCloudNum] = {0};
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
                        cube_updated[cubeInd] = true;
                    }
                }
                for (int i = 0; i < laserCloudValidNum; i++)
                {
                    int ind = laserCloudValidInd[i];

                    if(cube_updated[ind])
                    {
                        downSizeFilterMap.setInputCloud(featsArray[ind]);
                        downSizeFilterMap.filter(*featsArray[ind]);
                    }
                }
            #endif
            }
            t5 = omp_get_wtime();
                
            /******* Publish current frame points in world coordinates:  *******/
            laserCloudFullRes2->clear();
            *laserCloudFullRes2 = dense_map_en ? (*feats_undistort) : (* feats_down);

            int laserCloudFullResNum = laserCloudFullRes2->points.size();
    
            pcl::PointXYZI temp_point;
            laserCloudFullResColor->clear();
            {
            for (int i = 0; i < laserCloudFullResNum; i++)
            {
                RGBpointBodyToWorld(&laserCloudFullRes2->points[i], &temp_point);
                laserCloudFullResColor->push_back(temp_point);
            }

            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
            laserCloudFullRes3.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
            laserCloudFullRes3.header.frame_id = "/camera_init";
            pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }

            /******* Publish Effective points *******/
            {
            laserCloudFullResColor->clear();
            pcl::PointXYZI temp_point;
            for (int i = 0; i < laserCloudSelNum; i++)
            {
                RGBpointBodyToWorld(&laserCloudOri->points[i], &temp_point);
                laserCloudFullResColor->push_back(temp_point);
            }
            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
            laserCloudFullRes3.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
            laserCloudFullRes3.header.frame_id = "/camera_init";
            pubLaserCloudEffect.publish(laserCloudFullRes3);
            }

            /******* Publish Maps:  *******/
            sensor_msgs::PointCloud2 laserCloudMap;
            pcl::toROSMsg(*featsFromMap, laserCloudMap);
            laserCloudMap.header.stamp = ros::Time::now();//ros::Time().fromSec(last_timestamp_lidar);
            laserCloudMap.header.frame_id = "/camera_init";
            pubLaserCloudMap.publish(laserCloudMap);

            /******* Publish Odometry ******/
            odomAftMapped.header.frame_id = "/camera_init";
            odomAftMapped.child_frame_id = "/aft_mapped";
            odomAftMapped.header.stamp = ros::Time::now();//ros::Time().fromSec(last_timestamp_lidar);
            odomAftMapped.pose.pose.orientation.x = geoQuat.x;
            odomAftMapped.pose.pose.orientation.y = geoQuat.y;
            odomAftMapped.pose.pose.orientation.z = geoQuat.z;
            odomAftMapped.pose.pose.orientation.w = geoQuat.w;
            #ifdef USE_IKFOM
            odomAftMapped.pose.pose.position.x = state_point.pos(0);
            odomAftMapped.pose.pose.position.y = state_point.pos(1);
            odomAftMapped.pose.pose.position.z = state_point.pos(2);
            #else
            odomAftMapped.pose.pose.position.x = state.pos_end(0);
            odomAftMapped.pose.pose.position.y = state.pos_end(1);
            odomAftMapped.pose.pose.position.z = state.pos_end(2);
            #endif

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
            #ifdef USE_IKFOM
            msg_body_pose.pose.position.x = state_point.pos(0);
            msg_body_pose.pose.position.y = state_point.pos(1);
            msg_body_pose.pose.position.z = state_point.pos(2);
            #else
            msg_body_pose.pose.position.x = state.pos_end(0);
            msg_body_pose.pose.position.y = state.pos_end(1);
            msg_body_pose.pose.position.z = state.pos_end(2);
            #endif
            msg_body_pose.pose.orientation.x = geoQuat.x;
            msg_body_pose.pose.orientation.y = geoQuat.y;
            msg_body_pose.pose.orientation.z = geoQuat.z;
            msg_body_pose.pose.orientation.w = geoQuat.w;
            #ifdef DEPLOY
            mavros_pose_publisher.publish(msg_body_pose);
            #endif

            /******* Publish Path ********/
            msg_body_pose.header.frame_id = "/camera_init";
            path.poses.push_back(msg_body_pose);
            pubPath.publish(path);

            /*** save debug variables ***/
            frame_num ++;
            total_time_consu = t5 - t0;//aver_time_consu_predict * (frame_num - 1) / frame_num + (t5- t0) / frame_num;; //aver_time_consu_predict * (frame_num-1)/frame_num+t_propagate/frame_num;
            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
            
            // aver_time_consu = aver_time_consu * 0.8 + (t5 - t0) * 0.2;
            // T1.push_back(Measures.lidar_beg_time);
            // s_plot.push_back(aver_time_consu);
            // s_plot2.push_back(t5 - t3);
            // s_plot3.push_back(match_time);
            // s_plot4.push_back(float(feats_down_size/10000.0));
            // s_plot5.push_back(t5 - t0);

            T1.push_back(Measures.lidar_beg_time);
            s_averge_time.push_back(aver_time_consu);
            s_total_time.push_back(total_time_consu);

            #ifdef USE_IKFOM
            #ifdef Online_Calibration
            s_offt.push_back(state_point.offset_T_L_I);
            SO3 off_r_save = state_point.offset_R_L_I;
            vect3 euler_save =  SO3ToEuler(off_r_save);
            //Eigen::Vector3d euler_save = (R_offset.eulerAngles(0, 1, 2));
            s_offr.push_back(euler_save);
            #else
            vect3 null_vec;
            s_offt.push_back(null_vec);
            s_offr.push_back(null_vec);
            #endif
            //s_offr.push_back(vect3());
            s_vel.push_back(state_point.vel);
            SO3 r_save = state_point.rot;
            vect3 euler_save_r = SO3ToEuler(r_save);
            //Eigen::Vector3d euler_save_r = (R_global_cur.eulerAngles(0, 1, 2));
            s_rot.push_back(euler_save_r);
            s_pos.push_back(state_point.pos);
            s_bg.push_back(state_point.bg);
            s_ba.push_back(state_point.ba);
            s_grav.push_back(state_point.grav);
            //s_grav.push_back(grav);
            esekfom::esekf<state_ikfom, 12, input_ikfom>::cov cov_stat_cur = kf.get_P();
            #ifdef Online_Calibration
            double std_grav[2] = {std::sqrt(cov_stat_cur(21, 21)), std::sqrt(cov_stat_cur(22, 22))};
            double std_gravp[2] = {3*std::sqrt(cov_stat_cur(21, 21)), 3*std::sqrt(cov_stat_cur(22, 22))};
            double std_gravm[2] = {-3*std::sqrt(cov_stat_cur(21, 21)), -3*std::sqrt(cov_stat_cur(22, 22))};
            double std_gravp1[2] = {3*std::sqrt(cov_stat_cur(21, 21)), 0*std::sqrt(cov_stat_cur(22, 22))};
            double std_gravp2[2] = {0*std::sqrt(cov_stat_cur(21, 21)), 3*std::sqrt(cov_stat_cur(22, 22))};
            double std_gravm1[2] = {-3*std::sqrt(cov_stat_cur(21, 21)), 0*std::sqrt(cov_stat_cur(22, 22))};
            double std_gravm2[2] = {0*std::sqrt(cov_stat_cur(21, 21)), -3*std::sqrt(cov_stat_cur(22, 22))};
            #else
            double std_grav[2] = {std::sqrt(cov_stat_cur(15, 15)), std::sqrt(cov_stat_cur(16, 16))};
            double std_gravp[2] = {3*std::sqrt(cov_stat_cur(15, 15)), 3*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravm[2] = {-3*std::sqrt(cov_stat_cur(15, 15)), -3*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravp1[2] = {3*std::sqrt(cov_stat_cur(15, 15)), 0*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravp2[2] = {0*std::sqrt(cov_stat_cur(15, 15)), 3*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravm1[2] = {-3*std::sqrt(cov_stat_cur(15, 15)), 0*std::sqrt(cov_stat_cur(16, 16))};
            double std_gravm2[2] = {0*std::sqrt(cov_stat_cur(15, 15)), -3*std::sqrt(cov_stat_cur(16, 16))};
            #endif
		    vect2 delta(std_grav, 2);
            vect2 delta_cur;
            //S2 grav_con(0.31774, 0.209795, -9.80161);
            //S2 grav_con(-0.275336, -0.0900307, -9.80472);
            S2 grav_con(0.194428, 0.023413, -9.80795);
            grav_con.boxminus(delta_cur, state_point.grav);
		    vect2 deltap(std_gravp, 2);
		    vect2 deltap1(std_gravp1, 2);
		    vect2 deltap2(std_gravp2, 2);
		    vect2 deltam(std_gravm, 2);
		    vect2 deltam1(std_gravm1, 2);
		    vect2 deltam2(std_gravm2, 2);
		    //MTK::vectview<const double, 2> delta(std_grav, 2);
		    //delta(0) = std::sqrt(P_cov(42, 42));
		    //delta(1) = std::sqrt(P_cov(43, 43));
		    //vect3 exp_delta;
		    //exp_delta[0] = MTK::exp<double, 2>(exp_delta.template tail<2>(), delta); 
		    //exp_delta *= 9.8;
            S2 grav_plus(state_point.grav);
            S2 grav_plus1(state_point.grav);
            S2 grav_plus2(state_point.grav);
            S2 grav_minus(state_point.grav);
            S2 grav_minus1(state_point.grav);
            S2 grav_minus2(state_point.grav);
            grav_plus.boxplus(deltap);
            grav_plus1.boxplus(deltap1);
            grav_plus2.boxplus(deltap2);
            grav_minus.boxplus(deltam);
            grav_minus1.boxplus(deltam1); // for what?
            grav_minus2.boxplus(deltam2);
		    Eigen::Matrix<double, 3, 2> grav_matrix;
            #ifdef Online_Calibration
		    state_point.S2_Mx(grav_matrix, delta, 21);
            Eigen::Matrix<double, 3, 3> JcovJT = grav_matrix * cov_stat_cur.template block<2, 2>(21, 21) * grav_matrix.transpose();
            #else
		    state_point.S2_Mx(grav_matrix, delta, 15);
            Eigen::Matrix<double, 3, 3> JcovJT = grav_matrix * cov_stat_cur.template block<2, 2>(15, 15) * grav_matrix.transpose();
            #endif
		    // Eigen::Matrix<double, 3, 2> expx_x = Eigen::Matrix<state::scalar, 3, 2>::Zero();
		    // state_point.S2_expu_u(expx_x, delta, 15);
		    // vect3 exp_delta = grav_matrix * expx_x * delta;
            #ifdef Online_Calibration
            for(int i=0; i<3; i++)
		    {
			    pos_cov(i) = std::sqrt(cov_stat_cur(i, i));
			    rot_cov(i) = 3*std::sqrt(cov_stat_cur(i+3, i+3));
			    vel_cov(i) = std::sqrt(cov_stat_cur(12+i, 12+i));
                bg_cov(i) = std::sqrt(cov_stat_cur(15+i, 15+i));
                ba_cov(i) = std::sqrt(cov_stat_cur(18+i, 18+i));
			    offr_cov(i) = 3*std::sqrt(cov_stat_cur(6+i, 6+i));
			    offt_cov(i) = std::sqrt(cov_stat_cur(9+i, 9+i));
			    gra_cov(i) = std::sqrt(JcovJT(i, i)); // exp_delta(i);
                gra_plus(i) = grav_plus[i];
                gra_plus1(i) = grav_plus1[i];
                gra_plus2(i) = grav_plus2[i];
                gra_minus(i) = grav_minus[i];
                gra_minus1(i) = grav_minus1[i];
                gra_minus2(i) = grav_minus2[i];
		    }
            #else
		    for(int i=0; i<3; i++)
		    {
			    pos_cov(i) = std::sqrt(cov_stat_cur(i, i));
			    rot_cov(i) = 3*std::sqrt(cov_stat_cur(i+3, i+3));
			    vel_cov(i) = std::sqrt(cov_stat_cur(6+i, 6+i));
                bg_cov(i) = std::sqrt(cov_stat_cur(9+i, 9+i));
                ba_cov(i) = std::sqrt(cov_stat_cur(12+i, 12+i));
			    offr_cov(i) = 0.0; //std::sqrt(cov_stat_cur(6+i, 6+i));
			    offt_cov(i) = 0.0; //std::sqrt(cov_stat_cur(9+i, 9+i));
			    gra_cov(i) = std::sqrt(JcovJT(i, i)); // exp_delta(i);
                gra_plus(i) = grav_plus[i];
                gra_plus1(i) = grav_plus1[i];
                gra_plus2(i) = grav_plus2[i];
                gra_minus(i) = grav_minus[i];
                gra_minus1(i) = grav_minus1[i];
                gra_minus2(i) = grav_minus2[i];
		    }
            #endif
            SO3 rot_plus(state_point.rot);
            SO3 rot_minus(state_point.rot);
            rot_plus.boxplus(rot_cov);
            vect3 minus_rot_cov = -rot_cov;
            rot_minus.boxplus(minus_rot_cov);

            #ifdef Online_Calibration
            SO3 offr_plus(state_point.offset_R_L_I);
            SO3 offr_minus(state_point.offset_R_L_I);
            offr_plus.boxplus(offr_cov);
            vect3 minus_offr_cov = -offr_cov;
            offr_minus.boxplus(minus_offr_cov);
            #endif

		    s_cov_pos.push_back(pos_cov);
            vect3 euler_save_plus = SO3ToEuler(rot_plus);
            vect3 euler_save_minus = SO3ToEuler(rot_minus);
		    s_cov_rot_plus.push_back(euler_save_plus);
		    s_cov_rot_minus.push_back(euler_save_minus);
		    s_cov_vel.push_back(vel_cov);
		    s_cov_grav.push_back(gra_cov);
            #ifdef Online_Calibration
            vect3 euler_offr_plus = SO3ToEuler(offr_plus);
            vect3 euler_offr_minus = SO3ToEuler(offr_minus);
		    s_cov_offr_plus.push_back(euler_offr_plus);
		    s_cov_offr_minus.push_back(euler_offr_minus);
            #else
		    s_cov_offr_plus.push_back(null_vec);
		    s_cov_offr_minus.push_back(null_vec);
            #endif
		    s_cov_offt.push_back(offt_cov);
            s_cov_bg.push_back(bg_cov);
            s_cov_ba.push_back(ba_cov);
            s_grav_plus.push_back(gra_plus);
            s_grav_plus1.push_back(gra_plus1);
            s_grav_plus2.push_back(gra_plus2);
            s_grav_minus.push_back(gra_minus);
            s_grav_minus1.push_back(gra_minus1);
            s_grav_minus2.push_back(gra_minus2);
            s_grav_deltacur.push_back(delta_cur);
            s_grav_delta.push_back(delta);
            #else
            Eigen::Vector3d null_vec(0, 0, 0);
            s_offt.push_back(null_vec);
            s_offr.push_back(null_vec);
            s_vel.push_back(state.vel_end);
            s_pos.push_back(state.pos_end);
            #ifdef USE_QUA
            Eigen::Matrix3d rot_qua;
            // rot_qua<<ROT_FROM_QUA(state.qua_end);
            // Eigen::Vector3d euler_save = RotMtoEuler(rot_qua);
            Eigen::Quaternion<double> qua(state.qua_end[0], state.qua_end[1], state.qua_end[2], state.qua_end[3]);
            Eigen::Vector3d euler_save = SO3ToEuler_QUA(qua);
            #else
            Eigen::Vector3d euler_save = RotMtoEuler(state.rot_end);
            #endif
            s_rot.push_back(euler_save);
            s_bg.push_back(state.bias_g);
            s_ba.push_back(state.bias_a);
            s_grav.push_back(state.gravity);
            Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>  cov_stat_cur = state.cov;
            #ifdef USE_QUA
            s_cov_pos.push_back(null_vec);
		    s_cov_rot_plus.push_back(null_vec);
		    s_cov_rot_minus.push_back(null_vec);
		    s_cov_vel.push_back(null_vec);
		    s_cov_grav.push_back(null_vec);
		    s_cov_offr.push_back(null_vec);
		    s_cov_offt.push_back(null_vec);
            s_cov_bg.push_back(null_vec);
            s_cov_ba.push_back(null_vec);
            s_grav_plus.push_back(null_vec);
            s_grav_minus.push_back(null_vec);
            #else
            for(int i=0; i<3; i++)
            {
                pos_cov(i) = std::sqrt(cov_stat_cur(i+3, i+3));
			    rot_cov(i) = 3*std::sqrt(cov_stat_cur(i, i));
			    vel_cov(i) = std::sqrt(cov_stat_cur(6+i, 6+i));
                bg_cov(i) = std::sqrt(cov_stat_cur(9+i, 9+i));
                ba_cov(i) = std::sqrt(cov_stat_cur(12+i, 12+i));
			    offr_cov(i) = 0.0; //std::sqrt(cov_stat_cur(6+i, 6+i));
			    offt_cov(i) = 0.0; //std::sqrt(cov_stat_cur(9+i, 9+i));
			    gra_cov(i) = std::sqrt(cov_stat_cur(15+i, 15+i)); //std::sqrt(JcovJT(i, i)); // exp_delta(i);
            }
            Eigen::Matrix3d rot_plus = state.rot_end;
            Eigen::Matrix3d rot_minus = state.rot_end;
            Eigen::Matrix3d Exp_plus = Exp(rot_cov, 1.0);
            Eigen::Matrix3d Exp_minus = Exp(rot_cov, -1.0);
            rot_plus = state.rot_end * Exp_plus;
            rot_minus = state.rot_end * Exp_minus;

            Eigen::Vector3d euler_save_plus = RotMtoEuler(rot_plus);
            Eigen::Vector3d euler_save_minus = RotMtoEuler(rot_minus);
            Eigen::Vector3d gra_plus = state.gravity + 3 * gra_cov;
            Eigen::Vector3d gra_minus = state.gravity - 3 * gra_cov;

            s_cov_pos.push_back(pos_cov);
		    s_cov_rot_plus.push_back(euler_save_plus);
		    s_cov_rot_minus.push_back(euler_save_minus);
		    s_cov_vel.push_back(vel_cov);
		    s_cov_grav.push_back(gra_cov);
		    s_cov_offr.push_back(offr_cov);
		    s_cov_offt.push_back(offt_cov);
            s_cov_bg.push_back(bg_cov);
            s_cov_ba.push_back(ba_cov);
            s_grav_plus.push_back(gra_plus);
            s_grav_minus.push_back(gra_minus);
            #endif
            #endif
            // std::cout<<"[ mapping ]: time: copy map "<<copy_time<<" readd: "<<readd_time<<" match "<<match_time<<" solve "<<solve_time<<"acquire: "<<t4-t3<<" map incre "<<t5-t4<<" total "<<aver_time_consu<<std::endl;
            // fout_out << std::setw(10) << Measures.lidar_beg_time << " " << euler_cur.transpose()*57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose() \
            // <<" "<<state.bias_g.transpose()<<" "<<state.bias_a.transpose()<< std::endl;
            fout_out<<std::setw(8)<<laserCloudSelNum<<" "<<Measures.lidar_beg_time<<" "<<t2-t0<<" "<<match_time<<" "<<t5-t3<<" "<<t5-t0<<std::endl;
        }
        status = ros::ok();
        rate.sleep();
    }
    //--------------------------save map---------------
    // std::string surf_filename(map_file_path + "/surf.pcd");
    // std::string corner_filename(map_file_path + "/corner.pcd");
    // std::string all_points_filename(map_file_path + "/all_points.pcd");

    // PointCloudXYZI surf_points, corner_points;
    // surf_points = *featsFromMap;
    // fout_out.close();
    // fout_pre.close();
    // if (surf_points.size() > 0 && corner_points.size() > 0) 
    // {
    // pcl::PCDWriter pcd_writer;
    // std::cout << "saving...";
    // pcd_writer.writeBinary(surf_filename, surf_points);
    // pcd_writer.writeBinary(corner_filename, corner_points);
    // }

    if (!T1.empty())
    {
        #ifdef USE_IKFOM
        std::string  save_off = "/home/joanna-he/fast_lio1/data_save/estimate_off.csv";
        std::string save_cov="/home/joanna-he/fast_lio1/data_save/estimate_cov.csv";
        outputData(save_off, T1, s_offr, s_offt, s_vel, s_rot, s_pos, s_bg, s_ba, s_grav);
        outputCov(save_cov, s_cov_pos, s_cov_vel, s_cov_rot_plus, s_cov_rot_minus, s_cov_grav, s_cov_offr_plus, s_cov_offr_minus, s_cov_offt, s_cov_bg, s_cov_ba, s_grav_plus, s_grav_minus, s_grav_plus1, s_grav_minus1, s_grav_plus2, s_grav_minus2, s_grav_deltacur, s_grav_delta);
        #else
        std::string  save_off = "/home/joanna-he/fast_lio1/data_save/estimate_off_hand.csv";
        std::string save_cov="/home/joanna-he/fast_lio1/data_save/estimate_cov_hand.csv";
        outputData_state(save_off, T1, s_offr, s_offt, s_vel, s_rot, s_pos, s_bg, s_ba, s_grav);
        outputCov_state(save_cov, s_cov_pos, s_cov_vel, s_cov_rot_plus, s_cov_rot_minus, s_cov_grav, s_cov_offr, s_cov_offt, s_cov_bg, s_cov_ba, s_grav_plus, s_grav_minus);
        #endif
        std::string save_time="/home/joanna-he/fast_lio1/data_save/consumed_time.csv";
        outputAvetime(save_time, s_averge_time, s_total_time);

        #ifdef USE_IKFOM
        #ifdef RESTORE_VICON
        std::string save_truth="/home/joanna-he/fast_lio1/data_save/ground_truth.csv";
        outputGroundtruth(save_truth, v_time, pose_g, vel_g, rot_g);
        #endif
        #endif
        // plt::named_plot("total time",T1,s_plot5);
        // plt::named_plot("average time",T1,s_plot);
        // plt::named_plot("readd",T2,s_plot6);
        // plt::legend();
        // plt::show();
        // plt::pause(0.5);
        // plt::close();
    }
    else
    {
        std::cout << "no points saved";
    }
    return 0;
}
