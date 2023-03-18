#ifndef USE_IKFOM_CPP_
#define USE_IKFOM_CPP_

#include <use_ikfom.hpp>

MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::na, 0.0001);// 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::ng, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nba, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nbg, 0.00001);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 18, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 18, 1> res = Eigen::Matrix<double, 18, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
  // Eigen::Matrix3d Eye = ;
  vect3 f_pos = s.vel; // - MTK::hat(omega) * s.pos;
  vect3 f_vel = s.rot.toRotationMatrix() * (in.acc-s.ba); // - MTK::hat(omega) * s.vel;
//   vect3 f_vel = (in.acc-s.ba) - MTK::hat(omega) * s.vel;
//   vect3 f_rot = (s.rot.toRotationMatrix().transpose() - Eigen::Matrix3d::Identity()) * omega;
	for(int i = 0; i < 3; i++ ){
		res(i) = f_pos[i];
		res(i + 3) =  f_vel[i] + s.grav[i]; 
		res(i + 6) = omega[i]; // f_rot[i]; 
		// res(i + 15) = -omega[i]; 
	}
	return res;
}

// Eigen::Matrix<double, 18, 1> get_f_global(state_global &s, const input_ikfom &in)
// {
// 	Eigen::Matrix<double, 18, 1> res = Eigen::Matrix<double, 18, 1>::Zero();
// 	vect3 omega;
// 	in.gyro.boxminus(omega, s.bg);
//   // Eigen::Matrix3d Eye = ;
//   vect3 f_pos = s.vel;
//   vect3 f_vel = s.rot.toRotationMatrix() * (in.acc-s.ba);
//   vect3 f_rot = omega;
// 	for(int i = 0; i < 3; i++ ){
// 		res(i) = f_pos[i];
// 		res(i + 3) =  f_vel[i] + s.grav[i]; 
// 		res(i + 6) = f_rot[i]; 
// 	}
// 	return res;
// }

Eigen::Matrix<double, 18, 17> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 18, 17> cov = Eigen::Matrix<double, 18, 17>::Zero();
	cov.template block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	// cov.template block<3, 3>(0, 0) = -MTK::hat(omega);
	// cov.template block<3, 3>(0, 12) = -MTK::hat(s.pos);
	// cov.template block<3, 3>(3, 3) = -MTK::hat(omega);
	cov.template block<3, 3>(3, 6) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(3, 9) = -s.rot.toRotationMatrix(); // -Eigen::Matrix3d::Identity(); // 
	// cov.template block<3, 3>(3, 12) = -MTK::hat(s.vel);
	Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 15);
	cov.template block<3, 2>(3, 15) =  grav_matrix; 
  	// vect3 trans_omega = s.rot.toRotationMatrix().transpose() * omega;
	// cov.template block<3, 3>(6, 6) = MTK::hat(trans_omega); 
	cov.template block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity(); // - s.rot.toRotationMatrix().transpose(); 
	// cov.template block<3, 3>(15, 12) = Eigen::Matrix3d::Identity(); 
	return cov;
}

Eigen::Matrix<double, 18, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 18, 12> cov = Eigen::Matrix<double, 18, 12>::Zero();
	// cov.template block<3, 3>(0, 3) = -MTK::hat(s.pos);
	cov.template block<3, 3>(3, 0) = -s.rot.toRotationMatrix(); // -Eigen::Matrix3d::Identity(); // 
	// cov.template block<3, 3>(3, 3) = -MTK::hat(s.vel);
	cov.template block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity(); // - s.rot.toRotationMatrix().transpose(); // 
	cov.template block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
	// cov.template block<3, 3>(15, 3) = Eigen::Matrix3d::Identity();
	return cov;
}

vect3 SO3ToEuler(const SO3_type &orient) 
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
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}

#endif  // 
