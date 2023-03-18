// This is a modification of the algorithm described in the following paper:
//W.  Xu  and  F.  Zhang. Fast-lio:  A  fast,  robust  lidar-inertial  odome-try  package  by  tightly-coupled  iterated  kalman  filter. 
//arXiv  preprintarXiv:2010.08196, 2020

#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <fast_lio/States.h>
#include <fast_lio/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>


// #define DEBUG_PRINT
// #define USE_ikdtree

// #define USE_QUA
#ifdef USE_QUA
#define ON_CAL_QUA
#endif

#define USE_IKFOM

#define PI_M (3.14159265358)
#define G_m_s2 (9.8099)         // Gravaty const in GuangDong/China
#ifdef USE_QUA
#ifdef ON_CAL_QUA
#define DIM_OF_STATES (26)
#else
#define DIM_OF_STATES (19)
#endif
// typedef Eigen::Quaternion<double, Eigen::AutoAlign> quaternion;
#else
#define DIM_OF_STATES (18)      // Dimension of states (Let Dim(SO(3)) = 3)
#endif
#define DIM_OF_PROC_N (12)      // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN  (6.0)
#define LIDAR_SP_LEN    (2)
#define INIT_COV   (0.0001)

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define CONSTRAIN(v,min,max)     ((v>min)?((v<max)?v:max):min)
#define ARRAY_FROM_EIGEN(mat)    mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)  std::vector<decltype(mat)::Scalar> (mat.data(), mat.data() + mat.rows() * mat.cols())
#ifdef USE_QUA
#define VEC4_FROM_ARRAY(v) v[0], v[1], v[2], v[3]
#define ROT_FROM_QUA(v) v[0]*v[0]+v[1]*v[1]-v[2]*v[2]-v[3]*v[3], 2*(v[1]*v[2]-v[0]*v[3]), 2*(v[1]*v[3]+v[0]*v[2]),\
						2*(v[1]*v[2]+v[0]*v[3]), v[0]*v[0]-v[1]*v[1]+v[2]*v[2]-v[3]*v[3], 2*(v[2]*v[3]-v[0]*v[1]),\
						2*(v[1]*v[3]-v[0]*v[2]), 2*(v[2]*v[3]+v[0]*v[1]), v[0]*v[0]-v[1]*v[1]-v[2]*v[2]+v[3]*v[3];
#define F_X_QUA(v, n, dt) 1/n, -v[0]*dt/2/n, -v[1]*dt/2/n, -v[2]*dt/2/n,\
							v[0]*dt/2/n, 1/n, v[2]*dt/2/n, -v[1]*dt/2/n,\
							v[1]*dt/2/n, -v[2]*dt/2/n, 1/n, v[0]*dt/2/n,\
							v[2]*dt/2/n, v[1]*dt/2/n, -v[0]*dt/2/n, 1 /n;
#define F_X_QUA_BIAS_P1(v, n, dt) v[1]*dt/2/n, v[2]*dt/2/n, v[3]*dt/2/n,\
								-v[0]*dt/2/n, v[3]*dt/2/n, -v[2]*dt/2/n,\
								-v[3]*dt/2/n, -v[0]*dt/2/n, v[1]*dt/2/n,\
								v[2]*dt/2/n, -v[1]*dt/2/n, -v[0]*dt/2/n;
#define LEFT_MULTIPLY_QUA(v) -v[1], -v[2], -v[3],\
								v[0], -v[3], v[2],\
								v[3], v[0], -v[1],\
								-v[2], v[1], v[0];
#endif

#define DEBUG_FILE_DIR(name)  (std::string(std::string(ROOT_DIR) + "Log/"+ name))

typedef fast_lio::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;


Eigen::Matrix3d Eye3d(Eigen::Matrix3d::Identity());
Eigen::Matrix3f Eye3f(Eigen::Matrix3f::Identity());
Eigen::Vector3d Zero3d(0, 0, 0);
Eigen::Vector3f Zero3f(0, 0, 0);
Eigen::Vector4d Zero4d(0, 0, 0, 0);
Eigen::Vector4d Identity4d(1, 0, 0, 0);
// Eigen::Vector3d Lidar_offset_to_IMU(0.05512, 0.02226, 0.0297); // Horizon
Eigen::Vector3d Lidar_offset_to_IMU(0.04165, 0.02326, -0.0284); // Avia
Eigen::Vector3d Lid_offset_to_IMU(0.04165, 0.02326, -0.0284); // Avia
// Eigen::Matrix3d rotm_init=(Eigen::Matrix3d() << 0, 1, 0, -1, 0, 0, 0, 0, 1).finished();
Eigen::Matrix3d Lid_rot_to_IMU = Eigen::Matrix3d::Identity(); //rotm_init; //

struct MeasureGroup     // Lidar data and imu dates for the curent process
{
    MeasureGroup()
    {
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    PointCloudXYZI::Ptr lidar;
    std::deque<sensor_msgs::Imu::ConstPtr> imu;
};

struct StatesGroup
{
    StatesGroup() {
		#ifdef USE_QUA
		this->qua_end = Identity4d; // quaternion::Identity();
		#ifdef ON_CAL_QUA
		this->off_r = Identity4d;
		this->off_t = Zero3d;
		#endif
		#else
		this->rot_end = Eigen::Matrix3d::Identity();
		#endif
		this->pos_end = Zero3d;
        this->vel_end = Zero3d;
        this->bias_g  = Zero3d;
        this->bias_a  = Zero3d;
        this->gravity = Zero3d;
        this->cov     = Eigen::Matrix<double,DIM_OF_STATES,DIM_OF_STATES>::Identity() * INIT_COV;
		#ifdef USE_QUA
		this->cov(0,0)=this->cov(1,1)=this->cov(2,2)=this->cov(3,3)=this->cov(4,4)=this->cov(5,5)=this->cov(6,6)=this->cov(7,7)=this->cov(8,8)=this->cov(9,9)=1.0;
		#else
		this->cov(0,0)=this->cov(1,1)=this->cov(2,2)=this->cov(3,3)=this->cov(4,4)=this->cov(5,5)=this->cov(6,6)=this->cov(7,7)=this->cov(8,8)=1.0;
		#endif
	};

    StatesGroup(const StatesGroup& b) {
		#ifdef USE_QUA
		this->qua_end = b.qua_end;
		#ifdef ON_CAL_QUA
		this->off_r = b.off_r;
		this->off_t = b.off_t;
		#endif
		#else
		this->rot_end = b.rot_end;
		#endif
		this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g  = b.bias_g;
        this->bias_a  = b.bias_a;
        this->gravity = b.gravity;
        this->cov     = b.cov;
	};

    StatesGroup& operator=(const StatesGroup& b)
	{
		#ifdef USE_QUA
		this->qua_end = b.qua_end;
		#ifdef ON_CAL_QUA
		this->off_r = Identity4d;
		this->off_t = Zero3d;
		#endif
		#else
        this->rot_end = b.rot_end;
		#endif
		this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g  = b.bias_g;
        this->bias_a  = b.bias_a;
        this->gravity = b.gravity;
        this->cov     = b.cov;
        return *this;
	};


    StatesGroup operator+(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
	{
        StatesGroup a;
		#ifdef USE_QUA
		a.qua_end = (this->qua_end + state_add.block<4,1>(0,0))/(this->qua_end + state_add.block<4,1>(0,0)).norm(); //this->qua_end * quaternion(state_add(0,0), state_add(1,0), state_add(2,0), state_add(3,0))::normalize();
		a.pos_end = this->pos_end + state_add.block<3,1>(4,0);
        a.vel_end = this->vel_end + state_add.block<3,1>(7,0);
        a.bias_g  = this->bias_g  + state_add.block<3,1>(10,0);
        a.bias_a  = this->bias_a  + state_add.block<3,1>(13,0);
        a.gravity = this->gravity + state_add.block<3,1>(16,0);
		#ifdef ON_CAL_QUA
		a.off_r = (this->off_r + state_add.block<4,1>(19,0))/(this->off_r + state_add.block<4,1>(19,0)).norm();
		a.off_t = this->off_t + state_add.block<3,1>(23,0);
		#endif
		#else
		a.rot_end = this->rot_end * Exp(state_add(0,0), state_add(1,0), state_add(2,0));
		a.pos_end = this->pos_end + state_add.block<3,1>(3,0);
        a.vel_end = this->vel_end + state_add.block<3,1>(6,0);
        a.bias_g  = this->bias_g  + state_add.block<3,1>(9,0);
        a.bias_a  = this->bias_a  + state_add.block<3,1>(12,0);
        a.gravity = this->gravity + state_add.block<3,1>(15,0);
		#endif
        a.cov     = this->cov;
		return a;
	};

    StatesGroup& operator+=(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
	{
		#ifdef USE_QUA
		this->qua_end = (this->qua_end + state_add.block<4,1>(0,0))/(this->qua_end + state_add.block<4,1>(0,0)).norm(); //this->qua_end * quaternion(state_add(0,0), state_add(1,0), state_add(2,0), state_add(3,0))::normalize();
		this->pos_end += state_add.block<3,1>(4,0);
        this->vel_end += state_add.block<3,1>(7,0);
        this->bias_g  += state_add.block<3,1>(10,0);
        this->bias_a  += state_add.block<3,1>(13,0);
        this->gravity += state_add.block<3,1>(16,0);
		#ifdef ON_CAL_QUA
		this->off_r = (this->off_r + state_add.block<4,1>(19,0))/(this->off_r + state_add.block<4,1>(19,0)).norm();
		this->off_t += state_add.block<3,1>(23,0);
		#endif
		#else
        this->rot_end = this->rot_end * Exp(state_add(0,0), state_add(1,0), state_add(2,0));
		this->pos_end += state_add.block<3,1>(3,0);
        this->vel_end += state_add.block<3,1>(6,0);
        this->bias_g  += state_add.block<3,1>(9,0);
        this->bias_a  += state_add.block<3,1>(12,0);
        this->gravity += state_add.block<3,1>(15,0);
		#endif
		return *this;
	};

    Eigen::Matrix<double, DIM_OF_STATES, 1> operator-(const StatesGroup& b)
	{
        Eigen::Matrix<double, DIM_OF_STATES, 1> a;
		#ifdef USE_QUA
		a.block<4,1>(0,0) = this->qua_end - b.qua_end;
        a.block<3,1>(4,0)  = this->pos_end - b.pos_end;
        a.block<3,1>(7,0)  = this->vel_end - b.vel_end;
        a.block<3,1>(10,0)  = this->bias_g  - b.bias_g;
        a.block<3,1>(13,0) = this->bias_a  - b.bias_a;
        a.block<3,1>(16,0) = this->gravity - b.gravity;
		#ifdef ON_CAL_QUA
		a.block<4,1>(19,0) = this->off_r - b.off_r;
		a.block<3,1>(23,0) = this->off_t - b.off_t;
		#endif
		#else
        Eigen::Matrix3d rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3,1>(0,0)  = Log(rotd);
        a.block<3,1>(3,0)  = this->pos_end - b.pos_end;
        a.block<3,1>(6,0)  = this->vel_end - b.vel_end;
        a.block<3,1>(9,0)  = this->bias_g  - b.bias_g;
        a.block<3,1>(12,0) = this->bias_a  - b.bias_a;
        a.block<3,1>(15,0) = this->gravity - b.gravity;
		#endif
		return a;
	};

	#ifdef USE_QUA
	// quaternion qua_end;
	Eigen::Vector4d qua_end;
	#ifdef ON_CAL_QUA
	Eigen::Vector4d off_r;
	Eigen::Vector3d off_t;
	#endif
	#else
	Eigen::Matrix3d rot_end;      // the estimated attitude (rotation matrix) at the end lidar point
	#endif
    Eigen::Vector3d pos_end;      // the estimated position at the end lidar point (world frame)
    Eigen::Vector3d vel_end;      // the estimated velocity at the end lidar point (world frame)
    Eigen::Vector3d bias_g;       // gyroscope bias
    Eigen::Vector3d bias_a;       // accelerator bias
    Eigen::Vector3d gravity;      // the estimated gravity acceleration
    Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>  cov;     // states covariance
};

#ifdef USE_QUA
Eigen::Matrix<double, 3, 4> qua_vec_qua(Eigen::Vector4d &qua, Eigen::Vector3d &vec)
{
	Eigen::Matrix<double, 3, 4> res;
	Eigen::Vector3d quav(qua[1], qua[2], qua[3]);
	Eigen::Matrix3d qua_skew, vec_skew;
	qua_skew<<SKEW_SYM_MATRX(quav);
	vec_skew<<SKEW_SYM_MATRX(vec);
	Eigen::Vector3d col1 = qua[0] * vec + qua_skew*vec;
	Eigen::Matrix3d col234 = quav.transpose()*vec*Eye3d + quav*vec.transpose() - vec*quav.transpose() - qua[0]*vec_skew;
	res.block<3,1>(0,0) = 2 * col1;
	res.block<3,3>(0,1) = 2 * col234;
	return res;
}

Eigen::Matrix4d exp_q(Eigen::Vector3d &vec, double dt)
{
	Eigen::Matrix4d res, Omega;
	Eigen::Matrix3d vec_skew;
	vec_skew<<SKEW_SYM_MATRX(vec);
	Omega.block<1,3>(0,1) = -vec.transpose();
	Omega.block<3,1>(1,0) = vec;
	Omega.block<3,3>(1,1) = -vec_skew;
	res = Eigen::Matrix4d::Identity() + 0.5 * dt * Omega;
	return res;
}

Eigen::Vector3d SO3ToEuler_QUA(const Eigen::Quaternion<double> &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		Eigen::Vector3d euler_ang;
		euler_ang<<temp[0], temp[1], temp[2];
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		Eigen::Vector3d euler_ang;
		euler_ang<<temp[0], temp[1], temp[2];
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	Eigen::Vector3d euler_ang;
	euler_ang<<temp[0], temp[1], temp[2];
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}
#endif

template<typename T>
T rad2deg(T radians)
{
  return radians * 180.0 / PI_M;
}

template<typename T>
T deg2rad(T degrees)
{
  return degrees * PI_M / 180.0;
}

template<typename T>
auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g, \
                const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    // Eigen::Map<Eigen::Matrix3d>(rot_kp.rot, 3,3) = R;
    return std::move(rot_kp);
}

void outputAvetime(std::ostream& out,  std::vector<double>&t_average, std::vector<double> &t_total) 
{
	typename std::vector<double>::iterator it = t_average.begin();
	int i = 0;
	std::string compare[] = {"average", "total"};
	
	for(int j=0; j<2; j++){
			out << compare[j] << ',';
		//}
	}
	out << std::endl;
	for (; it != t_average.end(); it++, i++) {
		out << *it << ',' << t_total[i] << std::endl;
	}
}
void outputAvetime(const std::string& estimated, std::vector<double> &t_average, std::vector<double> &t_total) 
{
	std::ofstream out(estimated.c_str());
	outputAvetime(out, t_average, t_total);
	out.close();
}

void outputData_state(std::ostream& out,  std::vector<double> &time, std::vector<Eigen::Vector3d>&off_R, std::vector<Eigen::Vector3d> &off_T, std::vector<Eigen::Vector3d> &vel_global, std::vector<Eigen::Vector3d> &rot_global, \
									std::vector<Eigen::Vector3d> &pos_global, std::vector<Eigen::Vector3d> &bg, std::vector<Eigen::Vector3d> &ba, std::vector<Eigen::Vector3d> &grav) 
					{
	typename std::vector<Eigen::Vector3d>::iterator it = off_R.begin();
	int i = 0;
	std::string compare[] = {"euler_", "transition_", "vel_"};
	out << "time" << ",";
	std::string statedata[] = {"euler_", "transition_", "vel_", "rotation_", "position_", "bias_g_", "bias_a_", "gravity_"};
	for(int j=0; j<8; j++){
		for(int k=1; k<4; k++){
			out << statedata[j] << k << ',';
		}
	}
	out << std::endl;
	for (; it != off_R.end(); it++, i++) {
		out << time[i] << "," ; 
		for(int j=0; j < 3; j++){
		out << off_R[i][j] << ",";
		} 
		for(int j=0; j < 3; j++){
		out << off_T[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << vel_global[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << rot_global[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << pos_global[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << bg[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << ba[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << grav[i][j] << ",";
		}
		out << std::endl; 
	}
}
void outputData_state(const std::string& estimated, std::vector<double> &time, std::vector<Eigen::Vector3d> &off_R, std::vector<Eigen::Vector3d> &off_T, std::vector<Eigen::Vector3d> &vel_global, std::vector<Eigen::Vector3d> &rot_global, \
									std::vector<Eigen::Vector3d> &pos_global, std::vector<Eigen::Vector3d> &bg, std::vector<Eigen::Vector3d> &ba, std::vector<Eigen::Vector3d> &grav) 
{
	std::ofstream out(estimated.c_str());
	outputData_state(out, time, off_R, off_T, vel_global, rot_global, pos_global, bg, ba, grav);
	out.close();
}

void outputCov_state(std::ostream& out,  std::vector<Eigen::Vector3d>&cov_p, std::vector<Eigen::Vector3d> &cov_v, std::vector<Eigen::Vector3d> &cov_r_plus, std::vector<Eigen::Vector3d> &cov_r_minus, std::vector<Eigen::Vector3d> &cov_g, std::vector<Eigen::Vector3d> &cov_offr, std::vector<Eigen::Vector3d> &cov_offt, std::vector<Eigen::Vector3d> &cov_bg,  std::vector<Eigen::Vector3d> &cov_ba, std::vector<Eigen::Vector3d> &gra_plus, std::vector<Eigen::Vector3d> &gra_minus) 
{
	typename std::vector<Eigen::Vector3d>::iterator it = cov_p.begin();
	int i = 0;
	std::string compare[] = {"pos_", "vel_", "rotp_", "rotm_", "gra_", "offr_", "offt_", "bg_", "ba_", "grap_", "gram_"};
	for(int j=0; j<11; j++){
		for(int k=1; k<4; k++){
			out << compare[j] << k << ',';
		}
	}

	out << std::endl;
	for (; it != cov_p.end(); it++, i++) {
		for(int j=0; j < 3; j++){
		out << cov_p[i][j] << ",";
		} 
		for(int j=0; j < 3; j++){
		out << cov_v[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_r_plus[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_r_minus[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_g[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_offr[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_offt[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_bg[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_ba[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << gra_plus[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << gra_minus[i][j] << ",";
		}
		out << std::endl; 
	}
}
void outputCov_state(const std::string& estimated, std::vector<Eigen::Vector3d> &cov_p, std::vector<Eigen::Vector3d> &cov_v, std::vector<Eigen::Vector3d> &cov_r_plus, std::vector<Eigen::Vector3d> &cov_r_minus, std::vector<Eigen::Vector3d> &cov_g, std::vector<Eigen::Vector3d> &cov_offr, std::vector<Eigen::Vector3d> &cov_offt, std::vector<Eigen::Vector3d> &cov_bg, std::vector<Eigen::Vector3d> &cov_ba, std::vector<Eigen::Vector3d> &gra_plus, std::vector<Eigen::Vector3d> &gra_minus)
{
	std::ofstream out(estimated.c_str());
	outputCov_state(out, cov_p, cov_v, cov_r_plus, cov_r_minus, cov_g, cov_offr, cov_offt, cov_bg, cov_ba, gra_plus, gra_minus);
	out.close();
}

#endif
