// This is a modification of the algorithm described in the following paper:
//W.  Xu  and  F.  Zhang. Fast-lio:  A  fast,  robust  lidar-inertial  odome-try  package  by  tightly-coupled  iterated  kalman  filter. 
//arXiv  preprintarXiv:2010.08196, 2020

#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <fast_lio/States.h>
#include <geometry_msgs/Vector3.h>

#ifdef USE_IKFOM
#include "use-ikfom.hpp"
#endif

/// *************Preconfiguration

#define MAX_INI_COUNT (200)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  #ifdef USE_IKFOM
  Eigen::Matrix<double, 12, 12> Q = process_noise_cov();
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);
  #else
  void Process(const MeasureGroup &meas, StatesGroup &state, PointCloudXYZI::Ptr pcl_un_);
  #endif

  void Reset();
  #ifdef USE_IKFOM
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);
  #else
  void IMU_Initial(const MeasureGroup &meas, StatesGroup &state, int &N);
  void UndistortPcl(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_in_out);
  #endif
  // Eigen::Matrix3d Exp(const Eigen::Vector3d &ang_vel, const double &dt);

  void IntegrateGyr(const std::vector<sensor_msgs::Imu::ConstPtr> &v_imu);


  ros::NodeHandle nh;

  void Integrate(const sensor_msgs::ImuConstPtr &imu);
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);

  double scale_gravity;

  Eigen::Vector3d angvel_last;
  Eigen::Vector3d acc_s_last;

  Eigen::Matrix<double,DIM_OF_PROC_N,1> cov_proc_noise;

  Eigen::Vector3d cov_acc;
  Eigen::Vector3d cov_gyr;

  std::ofstream fout;

 private:
  /*** Whether is the first frame, init for first frame ***/
  bool b_first_frame_ = true;
  bool imu_need_init_ = true;

  int init_iter_num = 1;
  Eigen::Vector3d mean_acc;
  Eigen::Vector3d mean_gyr;

  /*** Undistorted pointcloud ***/
  PointCloudXYZI::Ptr cur_pcl_un_;

  //// For timestamp usage
  sensor_msgs::ImuConstPtr last_imu_;

  /*** For gyroscope integration ***/
  double start_timestamp_;
  /// Making sure the equal size: v_imu_ and v_rot_
  std::deque<sensor_msgs::ImuConstPtr> v_imu_;
  std::vector<Eigen::Matrix3d> v_rot_pcl_;
  std::vector<Pose6D> IMUpose;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), last_imu_(nullptr), start_timestamp_(-1)
{
  // #ifdef USE_IKFOM
  // Q = process_noise_cov();
  // #endif
  Eigen::Quaterniond q(0, 1, 0, 0);
  Eigen::Vector3d t(0, 0, 0);
  init_iter_num = 1;
  scale_gravity = 1.0;
  cov_acc       = Eigen::Vector3d(0.1, 0.1, 0.1);
  cov_gyr       = Eigen::Vector3d(0.1, 0.1, 0.1);
  mean_acc      = Eigen::Vector3d(0, 0, -1.0);
  mean_gyr      = Eigen::Vector3d(0, 0, 0);
  angvel_last   = Zero3d;
  cov_proc_noise = Eigen::Matrix<double,DIM_OF_PROC_N,1>::Zero();
  // Lidar_offset_to_IMU = Eigen::Vector3d(0.0, 0.0, -0.0);
  // fout.open(DEBUG_FILE_DIR("imu.txt"),std::ios::out);
}

ImuProcess::~ImuProcess() {fout.close();}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  scale_gravity  = 1.0;
  angvel_last   = Zero3d;
  cov_proc_noise = Eigen::Matrix<double,DIM_OF_PROC_N,1>::Zero();

  cov_acc   = Eigen::Vector3d(0.1, 0.1, 0.1);
  cov_gyr   = Eigen::Vector3d(0.1, 0.1, 0.1);
  mean_acc  = Eigen::Vector3d(0, 0, -1.0);
  mean_gyr  = Eigen::Vector3d(0, 0, 0);

  imu_need_init_ = true;
  b_first_frame_ = true;
  init_iter_num  = 1;

  last_imu_      = nullptr;

  //gyr_int_.Reset(-1, nullptr);
  start_timestamp_ = -1;
  v_imu_.clear();
  IMUpose.clear();

  cur_pcl_un_.reset(new PointCloudXYZI());
  fout.close();

}

#ifdef USE_IKFOM
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);

  Eigen::Vector3d cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    scale_gravity += (cur_acc.norm() - scale_gravity) / N;
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);


    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  // init_state.offset_T_L_I = Lid_offset_to_IMU;
  // init_state.offset_R_L_I = Lid_rot_to_IMU;
  kf_state.change_x(init_state);

  // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = Eigen::Matrix<double, 23, 23>::Identity() * 0.001;
  // kf_state.change_P(init_P);
}
#else
void ImuProcess::IMU_Initial(const MeasureGroup &meas, StatesGroup &state_inout, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  Eigen::Vector3d cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    scale_gravity += (cur_acc.norm() - scale_gravity) / N;
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    N ++;
  }

  state_inout.gravity = - mean_acc /scale_gravity * G_m_s2;
  #ifdef USE_QUA
  state_inout.qua_end = Identity4d;
  #ifdef ON_CAL_QUA
  state_inout.off_r = Identity4d;
  state_inout.off_t = Lidar_offset_to_IMU;
  #endif
  #else
  state_inout.rot_end = Eye3d; // Exp(mean_acc.cross(Eigen::Vector3d(0, 0, -1 / scale_gravity)));
  #endif
  state_inout.bias_g  = mean_gyr;
}
#endif

#ifdef USE_IKFOM
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1622179354; // - 1621800000;
  const double &imu_end_time = v_imu.back()->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1622179354; // - 1621800000;
  const double &pcl_beg_time = meas.lidar_beg_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  std::cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
           <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<std::endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(imu_beg_time-pcl_beg_time, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov check_P = kf_state.get_P();
  /*** forward propagation at each imu point ***/
  Eigen::Vector3d angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  Eigen::Matrix3d R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < pcl_beg_time)    continue; // - 1621800000  - 1622378400
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr     = acc_avr * G_m_s2; // / acc_avr.norm(); // mean_acc.norm(); // - state_inout.ba;

    if(head->header.stamp.toSec() < pcl_beg_time) // - 1621800000   - 1622378400
    {
      dt = tail->header.stamp.toSec() - pcl_beg_time; // - 1621800000  - 1622378400
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr * 10000;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc * 10000;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time; //- 1621800000  - 1621436000  - 1622378400
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  // dt = note * (pcl_end_time - imu_end_time);
  // kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();
  
  #ifdef DEBUG_PRINT
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P = kf_state.get_P();
    cout<<"[ IMU Process ]: vel "<<imu_state.vel.transpose()<<" pos "<<imu_state.pos.transpose()<<" ba"<<imu_state.ba.transpose()<<" bg "<<imu_state.bg.transpose()<<endl;
    cout<<"propagated cov: "<<P.diagonal().transpose()<<endl;
  #endif

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      Eigen::Matrix3d R_i(R_imu * Exp(angvel_avr, dt));
      
      Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      Eigen::Vector3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      #ifdef Online_Calibration
      Eigen::Vector3d P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);
      #else
      Eigen::Vector3d P_compensate = (imu_state.rot.conjugate() * (R_i * (P_i + Lid_offset_to_IMU) + T_ei) - Lid_offset_to_IMU);
      #endif      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
#else
void ImuProcess::UndistortPcl(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1621436000; //- 1621800000;
  const double &imu_end_time = v_imu.back()->header.stamp.toSec(); // - 1622378400; // - 1621436000; // - 1621436000; //- 1621800000;
  const double &pcl_beg_time = meas.lidar_beg_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  std::cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
           <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<std::endl;

  /*** Initialize IMU pose ***/
  IMUpose.clear();
  // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end, state.pos_end, state.rot_end));
  #ifdef USE_QUA
  Eigen::Matrix3d rot_end;
  rot_end << ROT_FROM_QUA(state_inout.qua_end);
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, rot_end));
  #else
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));
  #endif
  /*** forward propagation at each imu point ***/
  Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  #ifdef USE_QUA
  Eigen::Vector4d Q_imu(state_inout.qua_end);
  Eigen::Matrix3d R_imu;
  #else
  Eigen::Matrix3d R_imu(state_inout.rot_end);
  #endif
  Eigen::MatrixXd F_x(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity());
  Eigen::MatrixXd F_w(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_PROC_N>::Zero());
  Eigen::MatrixXd cov_w(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Zero());
  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu != (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    angvel_avr -= state_inout.bias_g;
    acc_avr     = acc_avr * G_m_s2 / scale_gravity - state_inout.bias_a;

    #ifdef DEBUG_PRINT
    // fout<<head->header.stamp.toSec()<<" "<<angvel_avr.transpose()<<" "<<acc_avr.transpose()<<std::endl;
    #endif  
    dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    
    /* covariance propagation */
    #ifdef USE_QUA
    // double q_norm = std::sqrt(1+dt*dt*angvel_avr.norm()*angvel_avr.norm()/4);
    Eigen::Matrix4d qua_exp = exp_q(angvel_avr, dt);
    // qua_diff<<F_X_QUA(angvel_avr, q_norm, dt);
    // Eigen::Matrix<double, 4, 3> qua_bg_p1;
    // qua_bg_p1<<F_X_QUA_BIAS_P1(state_inout.qua_end, q_norm, dt);
    Eigen::Matrix<double, 4, 3> qua_bg_p2;
    qua_bg_p2<<LEFT_MULTIPLY_QUA(state_inout.qua_end);
    // Eigen::Matrix<double, 4, 3> qua_bg = qua_bg_p1+dt*dt/4/q_norm/q_norm/q_norm*(state_inout.qua_end+qua_bg_p2*angvel_avr/2)*angvel_avr.transpose();
    F_x.block<4, 4>(0,0) = qua_exp;
    F_x.block<4, 3>(0,10) = -0.5 * dt * qua_bg_p2;
    // F_x.block<3, 3>(4,4) = Eye3d;
    F_x.block<3, 3>(4,7) = Eye3d * dt;
    // F_x.block<3, 3>(7,7) = Eye3d;
    F_x.block<3, 4>(7,0) = qua_vec_qua(state_inout.qua_end, acc_avr) * dt;
    Eigen::Matrix3d rot_qua;
    rot_qua<<ROT_FROM_QUA(state_inout.qua_end);
    F_x.block<3, 3>(7, 13) = -dt * rot_qua;
    F_x.block<3, 3>(7, 16) = Eye3d * dt;
    // F_x.block<9, 9>(10, 10) = Eigen::Matrix<double, 9, 9>::Identity();
    F_w.block<4, 3>(0,0) = -0.5 * dt * qua_bg_p2; //qua_bg;
    F_w.block<3, 3>(7,3) = -dt * rot_qua;
    F_w.block<3, 3>(10,6) = Eye3d * dt;
    F_w.block<3, 3>(13,9) = Eye3d * dt;

    Eigen::Matrix<double,DIM_OF_PROC_N,DIM_OF_PROC_N> cov_Q = Eigen::Matrix<double,DIM_OF_PROC_N,DIM_OF_PROC_N>::Zero();
    cov_Q.block<3, 3>(0,0).diagonal() = cov_gyr * 10000;
    cov_Q.block<3, 3>(3,3).diagonal() = cov_acc * 10000;
    cov_Q.block<3, 3>(6,6).diagonal() = Eigen::Vector3d(0.0001, 0.0001, 0.0001);
    cov_Q.block<3, 3>(9,0).diagonal() = Eigen::Vector3d(0.0001, 0.0001, 0.0001);

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + F_w*cov_Q*F_w.transpose();


    acc_imu = rot_qua * acc_avr + state_inout.gravity;

    state_inout.qua_end = qua_exp * state_inout.qua_end; //(state_inout.qua_end+qua_bg_p2*angvel_avr/2); ///q_norm;
    double q_norm = state_inout.qua_end.norm();
    state_inout.qua_end[0] /= q_norm;
    state_inout.qua_end[1] /= q_norm;
    state_inout.qua_end[2] /= q_norm;
    state_inout.qua_end[3] /= q_norm;
    state_inout.vel_end = state_inout.vel_end + acc_imu * dt; // rot_qua * acc_avr * dt + state_inout.gravity * dt;
    state_inout.pos_end = state_inout.pos_end + state_inout.vel_end * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last  = acc_imu;
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time; // - 1621800000  - 1622378400 
    // std::cout<<"acc "<<acc_imu.transpose()<<"vel "<<acc_imu.transpose()<<"vel "<<pos_imu.transpose()<<std::endl;
    Eigen::Matrix3d rot_q_cur;
    rot_q_cur<<ROT_FROM_QUA(state_inout.qua_end);
    IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, rot_q_cur));
    #else
    Eigen::Matrix3d acc_avr_skew;
    Eigen::Matrix3d Exp_f   = Exp(angvel_avr, dt);
    acc_avr_skew<<SKEW_SYM_MATRX(angvel_avr);

    F_x.block<3,3>(0,0)  = Exp(angvel_avr, - dt);
    F_x.block<3,3>(0,9)  = - Eye3d * dt;
    // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
    F_x.block<3,3>(3,6)  = Eye3d * dt;
    F_x.block<3,3>(6,0)  = - R_imu * acc_avr_skew * dt;
    F_x.block<3,3>(6,12) = - R_imu * dt;
    F_x.block<3,3>(6,15) = Eye3d * dt;

    Eigen::Matrix3d cov_acc_diag(Eye3d), cov_gyr_diag(Eye3d);
    cov_acc_diag.diagonal() = cov_acc;
    cov_gyr_diag.diagonal() = cov_gyr;
    cov_w.block<3,3>(0,0).diagonal()   = cov_gyr * dt * dt * 10000;
    cov_w.block<3,3>(3,3)              = R_imu * cov_gyr_diag * R_imu.transpose() * dt * dt * 10000;
    cov_w.block<3,3>(6,6)              = R_imu * cov_acc_diag * R_imu.transpose() * dt * dt * 10000;
    cov_w.block<3,3>(9,9).diagonal()   = Eigen::Vector3d(0.0001, 0.0001, 0.0001) * dt * dt; // bias gyro covariance
    cov_w.block<3,3>(12,12).diagonal() = Eigen::Vector3d(0.0001, 0.0001, 0.0001) * dt * dt; // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
    /* propogation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propogation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last  = acc_imu;
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time; // - 1621800000  - 1622378400
    // std::cout<<"acc "<<acc_imu.transpose()<<"vel "<<acc_imu.transpose()<<"vel "<<pos_imu.transpose()<<std::endl;
    IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    #endif
  }

  #ifndef USE_QUA
  /*** calculated the pos and attitude prediction at the frame-end ***/
  dt = pcl_end_time - imu_end_time;
  state_inout.vel_end = vel_imu + acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(angvel_avr, dt);
  state_inout.pos_end = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
  auto pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lidar_offset_to_IMU;
  #else
  Eigen::Matrix3d rot_q_cur;
  rot_q_cur<<ROT_FROM_QUA(state_inout.qua_end);
  #ifdef ON_CAL_QUA
  auto pos_liD_e = state_inout.pos_end + rot_q_cur * state_inout.off_t;
  #else
  auto pos_liD_e = state_inout.pos_end + rot_q_cur * Lidar_offset_to_IMU;
  #endif
  #endif
  // auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

  #ifdef DEBUG_PRINT
    std::cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos "<<state_inout.pos_end.transpose()<<" ba"<<state_inout.bias_a.transpose()<<" bg "<<state_inout.bias_g.transpose()<<std::endl;
    std::cout<<"propagated cov: "<<state_inout.cov.diagonal().transpose()<<std::endl;
  #endif

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    #ifdef USE_QUA
    R_imu<<MAT_FROM_ARRAY(head->rot);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    // std::cout<<"head imu acc: "<<acc_imu.transpose()<<std::endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);
    #else
    R_imu<<MAT_FROM_ARRAY(head->rot);
    acc_imu<<VEC_FROM_ARRAY(head->acc);
    // std::cout<<"head imu acc: "<<acc_imu.transpose()<<std::endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    angvel_avr<<VEC_FROM_ARRAY(head->gyr);
    #endif

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      Eigen::Matrix3d R_i(R_imu * Exp(angvel_avr, dt));

      Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      #ifdef USE_QUA
      Eigen::Matrix3d rot_qua;
      rot_qua<<ROT_FROM_QUA(state_inout.qua_end);
      #ifdef ON_CAL_QUA
      Eigen::Matrix3d rot_off;
      rot_off<<ROT_FROM_QUA(state_inout.off_r);
      Eigen::Vector3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - state_inout.pos_end);
      Eigen::Vector3d P_compensate = rot_off.transpose() * (rot_qua.transpose() * (R_i * (rot_off * P_i + state_inout.off_t) + T_ei) - state_inout.off_t);
      #else
      Eigen::Vector3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lidar_offset_to_IMU - pos_liD_e);
      Eigen::Vector3d P_compensate = rot_qua.transpose() * (R_i * P_i + T_ei);
      #endif
      #else
      Eigen::Vector3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lidar_offset_to_IMU - pos_liD_e);
      Eigen::Vector3d P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);
      #endif

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
#endif

#ifdef USE_IKFOM
void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
    }

    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();

  // {
  //   static ros::Publisher pub_UndistortPcl =
  //       nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
  //   sensor_msgs::PointCloud2 pcl_out_msg;
  //   pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
  //   pcl_out_msg.header.stamp = ros::Time().fromSec(meas.lidar_beg_time);
  //   pcl_out_msg.header.frame_id = "/livox";
  //   pub_UndistortPcl.publish(pcl_out_msg);
  // }

  /// Record last measurements
  last_imu_   = meas.imu.back();

  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
#else
void ImuProcess::Process(const MeasureGroup &meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {std::cout<<"no imu data"<<std::endl;return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_Initial(meas, stat, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT)
    {
      imu_need_init_ = false;
      // std::cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<std::endl;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               stat.gravity[0], stat.gravity[1], stat.gravity[2], stat.bias_g[0], stat.bias_g[1], stat.bias_g[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
    }

    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
  UndistortPcl(meas, stat, *cur_pcl_un_);

  t2 = omp_get_wtime();

  // {
  //   static ros::Publisher pub_UndistortPcl =
  //       nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
  //   sensor_msgs::PointCloud2 pcl_out_msg;
  //   pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
  //   pcl_out_msg.header.stamp = ros::Time().fromSec(meas.lidar_beg_time);
  //   pcl_out_msg.header.frame_id = "/livox";
  //   pub_UndistortPcl.publish(pcl_out_msg);
  // }

  /// Record last measurements
  last_imu_   = meas.imu.back();

  t3 = omp_get_wtime();
  
  std::cout<<"[ IMU Process ]: Time: "<<t3 - t1<<std::endl;
}
#endif