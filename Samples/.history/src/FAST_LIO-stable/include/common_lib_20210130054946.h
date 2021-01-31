// This is a modification of the algorithm described in the following paper:
//W.  Xu  and  F.  Zhang. Fast-lio:  A  fast,  robust  lidar-inertial  odome-try  package  by  tightly-coupled  iterated  kalman  filter. 
//arXiv  preprintarXiv:2010.08196, 2020

#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <Exp_mat.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <fast_lio/States.h>
#include <fast_lio/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <esekfom/esekfom.hpp> // HE


// #define DEBUG_PRINT

#define PI_M (3.14159265358)
#define G_m_s2 (9.8099)         // Gravaty const in GuangDong/China
#define DIM_OF_STATES (23)  
#define LON_OF_STATES (24)    
#define DIM_OF_PROC_N (12)      // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN  (6.0)
#define LIDAR_SP_LEN    (2)
#define INIT_COV  (0.0001) 

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define QUA_FROM_ARRAY(v) v[3], v[0], v[1], v[2] 
#define CONSTRAIN(v,min,max)     ((v>min)?((v<max)?v:max):min)
#define ARRAY_FROM_EIGEN(mat)    mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)  std::vector<decltype(mat)::Scalar> (mat.data(), mat.data() + mat.rows() * mat.cols())

#define DEBUG_FILE_DIR(name)  (std::string(std::string(ROOT_DIR) + "Log/"+ name))

typedef fast_lio::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

Eigen::Matrix3d Eye3d(Eigen::Matrix3d::Identity());
Eigen::Matrix3f Eye3f(Eigen::Matrix3f::Identity());
Eigen::Vector3d Zero3d(0, 0, 0);
Eigen::Vector3f Zero3f(0, 0, 0);

// define submanifold type
typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98099, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

// build compound manifold
MTK_BUILD_MANIFOLD(state,
((vect3, pos))
((vect3, vel))
((SO3, rot))
((vect3, bg))
((vect3, ba))
((S2, grav))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
);

MTK_BUILD_MANIFOLD(input,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

// prepare process noise covariance
MTK::get_cov<process_noise>::type process_noise_cov()
{
	MTK::get_cov<process_noise>::type cov = MTK::get_cov<process_noise>::type::Zero();
	MTK::setDiagonal<process_noise, vect3, 0>(cov, &process_noise::ng, 0.0001);// 0.03
	MTK::setDiagonal<process_noise, vect3, 3>(cov, &process_noise::na, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise, vect3, 6>(cov, &process_noise::nbg, 0.001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise, vect3, 9>(cov, &process_noise::nba, 0.001);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

//supply system-specific process model
double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 24, 1> get_f(state &s, const input &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 6) =  omega[i]; 
		res(i + 3) = a_inertial[i] + s.grav[i]; 
	}
	return res;
}

//supply system-specific differentions of the process model
Eigen::Matrix<double, 24, 23> df_dx(state &s, const input &in)
{
	Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
	cov.template block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(3, 6) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(3, 12) = -s.rot.toRotationMatrix();
	Eigen::Matrix<state::scalar, 2, 1> vec = Eigen::Matrix<state::scalar, 2, 1>::Zero();
	Eigen::Matrix<state::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 15);
	cov.template block<3, 2>(3, 15) =  grav_matrix; 
	cov.template block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity(); 
	return cov;
}


Eigen::Matrix<double, 24, 12> df_dw(state &s, const input &in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	cov.template block<3, 3>(3, 3) = -s.rot.toRotationMatrix();
	cov.template block<3, 3>(6, 0) = -Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
	return cov;
}

// prepare ekf class
class  ekf{
    public:
    MTK::get_cov<process_noise>::type Q = process_noise_cov();
	esekfom::esekf<state, 12, input> kf;

	ekf(state &start_state, esekfom::esekf<state, 12,input>::cov &init_P)
	{	
		kf = esekfom::esekf<state, 12, input>(start_state, init_P);
		//kf.init(get_f, df_dx, df_dw);
	}

	ekf()
	{
		esekfom::esekf<state, 12, input>::cov init_P =esekfom::esekf<state, 12, input>::cov::Identity() * INIT_COV;
		state start_state;
		kf = esekfom::esekf<state, 12, input>(start_state, init_P);
		//kf.init(get_f, df_dx, df_dw);
	}

	void predict(double &dt, input &in)
	{
		kf.predict(dt, Q, in);
	}
};


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

template<typename T>
auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g, \
                const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 4, 1> &R) 
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
		rot_kp.rot[i] = R(i); 
    }
    rot_kp.rot[3] = R(3); 
    return std::move(rot_kp);
}

template<typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot)
{
    T sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < 1e-6;
    T x, y, z;
    if(!singular)
    {
        x = atan2(rot(2, 1), rot(2, 2));
        y = atan2(-rot(2, 0), sy);   
        z = atan2(rot(1, 0), rot(0, 0));  
    }
    else
    {    
        x = atan2(-rot(1, 2), rot(1, 1));    
        y = atan2(-rot(2, 0), sy);    
        z = 0;
    }
    Eigen::Matrix<T, 3, 1> ang(x, y, z);
    return ang;
}

// output estimate data
void outputData(std::ostream& out,  std::vector<double> &time, std::vector<vect3>&off_R, std::vector<vect3> &off_T, std::vector<vect3> &vel_global, std::vector<vect3> &rot_global, \
									std::vector<vect3> &pos_global, std::vector<vect3> &bg, std::vector<vect3> &ba, std::vector<vect3> &grav) 
					{
	typename std::vector<vect3>::iterator it = off_R.begin();
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
void outputData(const std::string& estimated, std::vector<double> &time, std::vector<vect3> &off_R, std::vector<vect3> &off_T, std::vector<vect3> &vel_global, std::vector<vect3> &rot_global, \
									std::vector<vect3> &pos_global, std::vector<vect3> &bg, std::vector<vect3> &ba, std::vector<vect3> &grav) 
{
	std::ofstream out(estimated.c_str());
	outputData(out, time, off_R, off_T, vel_global, rot_global, pos_global, bg, ba, grav);
	out.close();
}

vect3 SO3ToEuler(const SO3 &orient) {
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
			vect3 euler_ang(temp, 3);
			return euler_ang;
		}
		if (test < -0.49999*unit) { // singularity at south pole

			_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
			double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
			vect3 euler_ang(temp, 3);
			return euler_ang;
		}
		
		_ang <<
				std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
				std::asin (2*test/unit),
				std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);

		return euler_ang;
	}

void outputCov(std::ostream& out,  std::vector<vect3>&cov_p, std::vector<vect3> &cov_v, std::vector<vect3> &cov_r, std::vector<vect3> &cov_g, std::vector<vect3> &cov_offr, std::vector<vect3> &cov_offt, std::vector<vect3> &cov_bg,  std::vector<vect3> &cov_ba, std::vector<vect3> &gra_plus, std::vector<vect3> &gra_minus, std::vector<vect3> &gra_plus1, std::vector<vect3> &gra_minus1, std::vector<vect3> &gra_plus2, std::vector<vect3> &gra_minus2, std::vector<vect2> &del_cur, std::vector<vect2> &del) 
{
	typename std::vector<vect3>::iterator it = cov_p.begin();
	int i = 0;
	std::string compare[] = {"pos_", "vel_", "rot_", "gra_", "offr_", "offt_", "bg_", "ba_", "grap_", "gram_", "grap1_", "gram1_", "grap2_", "gram2_"};
	std::string twodim[] = {"deltacur_", "delta_"};
	for(int j=0; j<14; j++){
		for(int k=1; k<4; k++){
			out << compare[j] << k << ',';
		}
	}
	for(int j=0; j<2;j++){
		for(int k=1; k<3; k++){
			out << twodim[j] << k << ",";
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
		out << cov_r[i][j] << ",";
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
		for(int j=0; j < 3; j++){
		out << gra_plus1[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << gra_minus1[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << gra_plus2[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << gra_minus2[i][j] << ",";
		}
		for(int j=0; j < 2; j++){
		out << del_cur[i][j] << ",";
		}
		for(int j=0; j < 2; j++){
		out << del[i][j] << ",";
		}
		out << std::endl; 
	}
}
void outputCov(const std::string& estimated, std::vector<vect3> &cov_p, std::vector<vect3> &cov_v, std::vector<vect3> &cov_r, std::vector<vect3> &cov_g, std::vector<vect3> &cov_offr, std::vector<vect3> &cov_offt, std::vector<vect3> &cov_bg, std::vector<vect3> &cov_ba, std::vector<vect3> &gra_plus, std::vector<vect3> &gra_minus, std::vector<vect3> &gra_plus1, std::vector<vect3> &gra_minus1, std::vector<vect3> &gra_plus2, std::vector<vect3> &gra_minus2, std::vector<vect2> &del_cur, std::vector<vect2> &del) 
{
	std::ofstream out(estimated.c_str());
	outputCov(out, cov_p, cov_v, cov_r, cov_g, cov_offr, cov_offt, cov_bg, cov_ba, gra_plus, gra_minus, gra_plus1, gra_minus1, gra_plus2, gra_minus2, del_cur, del);
	out.close();
}

void outputAvetime(std::ostream& out,  std::vector<double>&t_update, std::vector<double> &t_predict) 
{
	typename std::vector<double>::iterator it = t_update.begin();
	int i = 0;
	std::string compare[] = {"update_average", "predict_average"};
	
	for(int j=0; j<2; j++){
			out << compare[j] << ',';
	}
	out << std::endl;
	for (; it != t_update.end(); it++, i++) {
		out << *it << ',' << t_predict[i] << std::endl;
	}
}
void outputAvetime(const std::string& estimated, std::vector<double> &t_update, std::vector<double> &t_predict) 
{
	std::ofstream out(estimated.c_str());
	outputAvetime(out, t_update, t_predict);
	out.close();
}

#endif
