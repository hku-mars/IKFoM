/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef USE_IKFOM_H
#define USE_IKFOM_H

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

#define Online_Calibration
#define RESTORE_VICON

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98099, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

#ifdef Online_Calibration
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos)) 
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((S2, grav))
);
#else
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos)) 
((SO3, rot))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((S2, grav))
);
#endif

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);// 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.003); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.003);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
#ifdef Online_Calibration
Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) =  omega[i]; 
		res(i + 12) = a_inertial[i] + s.grav[i]; 
	}
	return res;
}

Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
	Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 2>(12, 21) =  grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); 
	return cov;
}


Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
	return cov;
}
#else
Eigen::Matrix<double, 17, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 17, 1> res = Eigen::Matrix<double, 17, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) =  omega[i]; 
		res(i + 6) = a_inertial[i] + s.grav[i]; 
	}
	return res;
}

Eigen::Matrix<double, 17, 17> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 17, 17> cov = Eigen::Matrix<double, 17, 17>::Zero();
	cov.template block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(6, 3) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(6, 12) = -s.rot.toRotationMatrix();
	Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 15);
	cov.template block<3, 2>(6, 15) =  grav_matrix; 
	cov.template block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity(); 
	return cov;
}


Eigen::Matrix<double, 17, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 17, 12> cov = Eigen::Matrix<double, 17, 12>::Zero();
	cov.template block<3, 3>(6, 3) = -s.rot.toRotationMatrix();
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
	return cov;
}
#endif

vect3 SO3ToEuler(const SO3 &orient) 
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

void outputCov(std::ostream& out,  std::vector<vect3>&cov_p, std::vector<vect3> &cov_v, std::vector<vect3> &cov_r_plus, std::vector<vect3> &cov_r_minus, std::vector<vect3> &cov_g, std::vector<vect3> &cov_offr_plus, std::vector<vect3> &cov_offr_minus, std::vector<vect3> &cov_offt, std::vector<vect3> &cov_bg,  std::vector<vect3> &cov_ba, std::vector<vect3> &gra_plus, std::vector<vect3> &gra_minus, std::vector<vect3> &gra_plus1, std::vector<vect3> &gra_minus1, std::vector<vect3> &gra_plus2, std::vector<vect3> &gra_minus2, std::vector<vect2> &del_cur, std::vector<vect2> &del) 
{
	typename std::vector<vect3>::iterator it = cov_p.begin();
	int i = 0;
	std::string compare[] = {"pos_", "vel_", "rotp_", "rotm_", "gra_", "offrp_", "offrm_", "offt_", "bg_", "ba_", "grap_", "gram_", "grap1_", "gram1_", "grap2_", "gram2_"};
	std::string twodim[] = {"deltacur_", "delta_"};
	for(int j=0; j<16; j++){
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
		out << cov_r_plus[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_r_minus[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_g[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_offr_plus[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << cov_offr_minus[i][j] << ",";
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
void outputCov(const std::string& estimated, std::vector<vect3> &cov_p, std::vector<vect3> &cov_v, std::vector<vect3> &cov_r_plus, std::vector<vect3> &cov_r_minus, std::vector<vect3> &cov_g, std::vector<vect3> &cov_offr_plus, std::vector<vect3> &cov_offr_minus, std::vector<vect3> &cov_offt, std::vector<vect3> &cov_bg, std::vector<vect3> &cov_ba, std::vector<vect3> &gra_plus, std::vector<vect3> &gra_minus, std::vector<vect3> &gra_plus1, std::vector<vect3> &gra_minus1, std::vector<vect3> &gra_plus2, std::vector<vect3> &gra_minus2, std::vector<vect2> &del_cur, std::vector<vect2> &del) 
{
	std::ofstream out(estimated.c_str());
	outputCov(out, cov_p, cov_v, cov_r_plus, cov_r_minus, cov_g, cov_offr_plus, cov_offr_minus, cov_offt, cov_bg, cov_ba, gra_plus, gra_minus, gra_plus1, gra_minus1, gra_plus2, gra_minus2, del_cur, del);
	out.close();
}

#ifdef RESTORE_VICON
void outputGroundtruth(std::ostream& out, std::vector<double> &time, std::vector<vect3>&pos_g, std::vector<vect3> &vel_g, std::vector<vect3> &rot_g) 
					{
	typename std::vector<vect3>::iterator it = pos_g.begin();
	int i = 0;
	std::string compare[] = {"euler_", "transition_", "vel_"};
	out << "time" << ",";
	std::string statedata[] = {"pos_", "vel_", "rot_"};
	for(int j=0; j<3; j++){
		for(int k=1; k<4; k++){
			out << statedata[j] << k << ',';
		}
	}
	out << std::endl;
	for (; it != pos_g.end(); it++, i++) {
		out << time[i] << "," ; 
		for(int j=0; j < 3; j++){
		out << pos_g[i][j] << ",";
		} 
		for(int j=0; j < 3; j++){
		out << vel_g[i][j] << ",";
		}
		for(int j=0; j < 3; j++){
		out << rot_g[i][j] << ",";
		}
		out << std::endl; 
	}
}
void outputGroundtruth(const std::string& estimated, std::vector<double> &time, std::vector<vect3> &pos_g, std::vector<vect3> &vel_g, std::vector<vect3> &rot_g) 
{
	std::ofstream out(estimated.c_str());
	outputGroundtruth(out, time, pos_g, vel_g, rot_g);
	out.close();
}
#endif

#endif