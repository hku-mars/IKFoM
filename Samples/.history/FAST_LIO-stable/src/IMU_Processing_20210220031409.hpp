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
#include <Exp_mat.h>
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

/// *************Preconfiguration

#define MAX_INI_COUNT (200)
#define IMU_ACC_SCALE (9.81)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Process(const MeasureGroup &meas, ekf &state_cal, PointCloudXYZI::Ptr pcl_un_);
  void Reset();
  void IMU_Initial(const MeasureGroup &meas, ekf &state_cal, int &N);

  void IntegrateGyr(const std::vector<sensor_msgs::Imu::ConstPtr> &v_imu);

  void UndistortPcl(const MeasureGroup &meas, ekf &state_inout, PointCloudXYZI &pcl_in_out);

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

  start_timestamp_ = -1;
  v_imu_.clear();
  IMUpose.clear();

  cur_pcl_un_.reset(new PointCloudXYZI());
  fout.close();

}

void ImuProcess::IMU_Initial(const MeasureGroup &meas, ekf &state_inout, int &N)
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

  // initialize state
  state init_state = state_inout.kf.get_x();
  double ini_gyro[3] = {0.1, 0.1, 0.1} ; 
  double ini_acc[3] = {0.1, 0.1, 0.1} ; 
  init_state.bg = mean_gyr; 
  init_state.ba =  vect3(); 
  init_state.vel = vect3();
  init_state.pos = vect3();
  init_state.grav = S2(-mean_acc * G_m_s2); 
  init_state.offset_T_L_I = Lidar_offset_to_IMU;
  init_state.rot = SO3();
  state_inout.kf.change_x(init_state);

  //initialize covariance
  esekfom::esekf<state, 12, input>::cov init_P =esekfom::esekf<state, 12, input>::cov::Identity() * INIT_COV;
  
  init_P(9, 9) = 0.001;
  init_P(10, 10) = 0.001;
  init_P(11, 11) = 0.001;
  init_P(12, 12) = 0.01;
  init_P(13, 13) = 0.01;
  init_P(14, 14) = 0.01;
	
  state_inout.kf.change_P(init_P);
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, ekf &state_inout, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  state in_state = state_inout.kf.get_x();
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  std::cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
           <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<std::endl;

  /*** Initialize IMU pose ***/
  IMUpose.clear();
  Eigen::Matrix<double, 4, 1> q_data = in_state.rot.coeffs().transpose();
  IMUpose.push_back(set_pose6d(imu_beg_time - pcl_beg_time, Zero3d, Zero3d, in_state.vel, in_state.pos, q_data));  

  /*** forward propagation at each imu point ***/
  Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu, pos_imu; 
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


    #ifdef DEBUG_PRINT
    // fout<<head->header.stamp.toSec()<<" "<<angvel_avr.transpose()<<" "<<acc_avr.transpose()<<std::endl;
    #endif  
    dt = tail->header.stamp.toSec() - head->header.stamp.toSec();

    input in_cur;
    in_cur.acc = acc_avr * 9.81; 
    in_cur.gyro = angvel_avr; 
    state_inout.Q.diagonal().segment<3>(0) = cov_gyr  * 10000; 
    state_inout.Q.diagonal().segment<3>(3) = cov_acc  * 10000; 
    state_inout.predict(dt, in_cur);
   
    state cur_state = state_inout.kf.get_x();
    vect3 acc_g_avr= cur_state.rot*(acc_avr * 9.81 - cur_state.ba); 
     for(int i = 0; i < 3; i++){
    acc_imu[i] =  acc_g_avr[i] + cur_state.grav[i]; 
    }
    angvel_last = ( angvel_avr - cur_state.bg );
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    Eigen::Matrix<double, 4, 1> q_data_ = cur_state.rot.coeffs().transpose();
    IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_last, cur_state.vel, cur_state.pos, q_data_));
  }

  /*** undistort each lidar point (backward propagation) ***/
  state cur_state_ = state_inout.kf.get_x();
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;

    SO3 R_imu(QUA_FROM_ARRAY(head->rot));
    acc_imu<<VEC_FROM_ARRAY(it_kp->acc);
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    angvel_avr<<VEC_FROM_ARRAY(it_kp->gyr);

      int i = 0;
      for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
      {
        dt = it_pcl->curvature / double(1000) - head->offset_time;
      
      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
        Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        Eigen::Vector3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - cur_state_.pos);
        SO3 R_imu_k(R_imu);
        R_imu_k.boxplus(angvel_avr, dt);
        Eigen::Vector3d P_compensate = cur_state_.offset_R_L_I.conjugate() * (cur_state_.rot.conjugate() * (R_imu_k * (cur_state_.offset_R_L_I * P_i + cur_state_.offset_T_L_I) + T_ei) - cur_state_.offset_T_L_I);// not accurate!
        
      /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == pcl_out.points.begin()) break;
      }
  }
}

void ImuProcess::Process(const MeasureGroup &meas, ekf &stat, PointCloudXYZI::Ptr cur_pcl_un_)
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
    state state_info = stat.kf.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      imu_need_init_ = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               state_info.grav[0], state_info.grav[1], state_info.grav[2], state_info.bg[0], state_info.bg[1], state_info.bg[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
    }

    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
  UndistortPcl(meas, stat, *cur_pcl_un_);

  t2 = omp_get_wtime();

  /// Record last measurements
  last_imu_   = meas.imu.back();

  t3 = omp_get_wtime();
  
  std::cout<<"[ IMU Process ]: Time: "<<t3 - t1<<std::endl;
}