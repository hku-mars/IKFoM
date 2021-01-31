/*
 *  Copyright (c) 2008--2011, Universitaet Bremen
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

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP


#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <mtk/types/vect.hpp>
#include <mtk/types/SOn.hpp>
#include <mtk/types/S2.hpp>
#include <mtk/startIdx.hpp>
#include <mtk/build_manifold.hpp>
#include "util.hpp"

namespace esekfom {

using namespace Eigen;

struct dyn_cal
{
	bool v;
	bool t;
	Eigen::VectorXd h_;
	Eigen::VectorXd z_o;
	Eigen::MatrixXd h_v;
};

struct dyn_cal_modified
{
	bool v;
	bool t;
	Eigen::VectorXd h_;
};

template<typename state, int process_noise_dof, typename measurement=state, int measurement_noise_dof=0, typename input = state>
class esekf_{

	typedef esekf_ self;
	enum{
		n = state::DOF, m = state::DIM, l = measurement::DOF
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef Matrix<scalar_type, m, n> cov_;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;
	typedef Matrix<scalar_type, m, 1> flatted_state;
	typedef flatted_state processModel(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
	typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;
	typedef measurement measurementModel(state &, bool &);
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> measurementModel_dyn(state &, bool &);
	typedef Eigen::Matrix<scalar_type ,l, n> measurementMatrix1(state &, bool&);
	typedef Eigen::Matrix<scalar_type , Eigen::Dynamic, n> measurementMatrix1_dyn(state &, bool&);
	typedef Eigen::Matrix<scalar_type ,l, measurement_noise_dof> measurementMatrix2(state &, bool&);
	typedef Eigen::Matrix<scalar_type ,Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &, bool&);
	typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

	esekf_(const state &x = state(),
		const cov  &P = cov::Identity()): x_(x), P_(P){
		SparseMatrix<scalar_type> ref(n, n);
		ref.setIdentity();
		l_ = ref;
		f_x_2 = ref;
		f_x_1 = ref;
	};

	//template<typename processModel, typename processMatrix1, typename processMatrix2, typename measurementModel, typename measurementMatrix1, typename measurementMatrix2>
	void init(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel h_in, measurementMatrix1 h_x_in, measurementMatrix2 h_v_in)
	{
		f = f_in;
		f_x = f_x_in;
		f_w = f_w_in;
		h = h_in;
		h_x = h_x_in;
		h_v = h_v_in;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
	}

	//template<typename processModel, typename processMatrix1, typename processMatrix2>
	void init_dyn(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel_dyn h_in, measurementMatrix1_dyn h_x_in, measurementMatrix2_dyn h_v_in)
	{
		f = f_in;
		f_x = f_x_in;
		f_w = f_w_in;
		h_dyn = h_in;
		h_x_dyn = h_x_in;
		h_v_dyn = h_v_in;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
	}

	//template<typename processnoisecovariance>
	void predict(double &dt, processnoisecovariance &Q, const input &i_in){
		flatted_state f_ = f(x_, i_in);
		cov_ f_x_ = f_x(x_, i_in);
		cov f_x_final;

		Matrix<scalar_type, m, process_noise_dof> f_w_ = f_w(x_, i_in);
		Matrix<scalar_type, n, Eigen::Dynamic> f_w_final = Matrix<scalar_type, n, Eigen::Dynamic>::Zero(n, Q.cols());
		state x_before = x_;
		x_.oplus(f_, dt);

		F_x1 = cov::Identity();
		for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
			int idx = (*it).first.first;
			int dim = (*it).first.second;
			int dof = (*it).second;
			for(int i = 0; i < int(n); i++){
				for(int j=0; j<dof; j++)
				{f_x_final(idx+j, i) = f_x_(dim+j, i);}	
			}
			for(int i = 0; i < int(f_w_.cols()); i++){
				for(int j=0; j<dof; j++)
				{f_w_final(idx+j, i) = f_w_(dim+j, i);}
			}
		}
		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = -1 * f_(dim + i) * dt;
			}

			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
			F_x1.template block<3, 3>(idx, idx) = res.toRotationMatrix();
			
			res_temp_SO3 = MTK::A_matrix(seg_SO3);
			for(int i = 0; i < int(n); i++){
				f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < int(f_w_.cols()); i++){
				f_w_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_. template block<3, 1>(dim, i));
			}
		}
		
		
		Matrix<scalar_type, 2, 3> res_temp_S2;
		MTK::vect<3, scalar_type> seg_S2;
		for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_S2(i) = f_(dim + i) * dt;
			}
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			x_.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, vec, idx);
			F_x1.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;

			Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
			x_before.S2_hat(x_before_hat, idx);
			res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat*MTK::A_matrix(seg_S2).transpose();
			
			for(int i = 0; i < int(n); i++){
				f_x_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_x_. template block<3, 1>(dim, i));
				
			}
			for(int i = 0; i < int(f_w_.cols() ); i++){
				f_w_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_w_. template block<3, 1>(dim, i));
			}
		}
		
		F_x1 += f_x_final * dt;
		P_ = (F_x1) * P_ * (F_x1).transpose() + (dt * f_w_final) * Q * (dt * f_w_final).transpose();
	}

	//template<typename processnoisecovariance>
	void predict_sparse(double &dt, processnoisecovariance &Q, const input &i_in ){
		flatted_state f_ = f(x_, i_in);
		state x_before = x_;
		x_.oplus(f_, dt);
		cov_ f_x_ = f_x(x_, i_in);
		cov f_x_final;
		Matrix<scalar_type, m, Eigen::Dynamic> f_w_ = f_w(x_, i_in);
		Matrix<scalar_type, n, Eigen::Dynamic> f_w_final = Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, Q.cols());

		F_x1 = cov::Identity();
		for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
			int idx = (*it).first.first;
			int dim = (*it).first.second;
			int dof = (*it).second;
			for(int i = 0; i < int(n); i++){
				for(int j=0; j<dof; j++)
				{f_x_final(idx+j, i) = f_x_(dim+j, i);}	
			}
			for(int i = 0; i < int(f_w_.cols()); i++){ //f_w_.cols() = Q.cols_()
				for(int j=0; j<dof; j++)
				{f_w_final(idx+j, i) = f_w_(dim+j, i);}
			}
		}
		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) { 
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = -1 * f_(dim + i) * dt;
			}
			
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
			
			res_temp_SO3 = res.toRotationMatrix();
			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					f_x_1.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
				}
			}

			res_temp_SO3 = MTK::A_matrix(seg_SO3);
			for(int i = 0; i < int(n); i++){
				f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < int(f_w_.cols()); i++){
				f_w_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_. template block<3, 1>(dim, i));
			}
		}

		Matrix<scalar_type, 2, 3> res_temp_S2;
		Matrix<scalar_type, 2, 2> res_temp_S2_;
		MTK::vect<3, scalar_type> seg_S2;
		for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_S2(i) = f_(dim + i) * dt;
			}
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			x_.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, vec, idx);
			res_temp_S2_ = Nx * res.toRotationMatrix() * Mx;

			Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
			x_before.S2_hat(x_before_hat, idx);
			res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat*MTK::A_matrix(seg_S2).transpose();

			for(int i = 0; i < 2; i++){
				for(int j = 0; j < 2; j++){
					f_x_1.coeffRef(idx + i, idx + j) = res_temp_S2_(i, j);
				}
			}

			for(int i = 0; i < int(n); i++){
				f_x_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_x_. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < int(f_w_.cols() ); i++){
				f_w_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_w_. template block<3, 1>(dim, i));
			}
		}

		f_x_1.makeCompressed();
		spMt f_x2 = f_x_final.sparseView();
		spMt f_w1 = f_w_final.sparseView();
		spMt xp = f_x_1 + f_x2 * dt;
		P_ = xp * P_ * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
	}

	//template<typename measurementnoisecovariance>
	void update_iterated(measurement& z, measurementnoisecovariance &R, int maximun_iter, std::vector<double> &limit, double ampl) {
		
		if(!(is_same<typename measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}
		int t = 0;
		bool converg = true;   
		bool valid = true;    
		state x_propagated = x_;
		cov P_propagated = P_;
		static const int dof_Measurement = measurement::DOF;
		for(int i=0; i<maximun_iter; i++)
		{
			vectorized_state dx, dx_new;
			x_.boxminus(dx, x_propagated);
			dx_new = dx;
			Matrix<scalar_type, dof_Measurement, n> h_x_ = h_x(x_, valid);
			Matrix<scalar_type, dof_Measurement, Eigen::Dynamic> h_v_ = h_v(x_, valid);
			if(! valid)
			{
				continue; 
			}

			P_ = P_propagated;
			
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);
				}

				res_temp_SO3 = A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx.template block<3, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_propagated. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 3>(i, idx) =(P_propagated. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}
		
		
			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}
				
				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx.template block<2, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<2, 1>(idx, i) = res_temp_S2 * (P_propagated. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 2>(i, idx) = (P_propagated. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			Matrix<scalar_type, n, dof_Measurement> K_;
			if(n > dof_Measurement)
			{
				K_= P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_ * R * h_v_.transpose()).inverse();
			}
			else
			{
				//Eigen::Matrix<scalar_type, measurement_noise::DOF, measurement_noise::DOF> R_in = R.inverse();
				measurementnoisecovariance R_in = (h_v_*R*h_v_.transpose()).inverse();
				K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in; // * R_in;
			}
			Matrix<scalar_type, dof_Measurement, 1> innovation; 
			z.boxminus(innovation, h(x_, valid));
			cov K_x = K_ * h_x_;
			Matrix<scalar_type, n, 1> dx_ = K_ * innovation + (K_x  - Matrix<scalar_type, n, n>::Identity()) * dx_new;
        	state x_before = x_;
			x_.boxplus(dx_);

			converg = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) * ampl > limit[i])
				{
					converg = false;
					break;
				}
			}

			if(converg) t++;
	        
			if(t > 1 || i == maximun_iter - 1)
			{
				L_ = P_;
		
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = A_matrix(seg_SO3).transpose();
					for(int i = 0; i < int(n); i++){
						L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
					}
					if(n > dof_Measurement)
					{
						for(int i = 0; i < int(dof_Measurement); i++){
							K_. template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
						}
					}
					else
					{
						for(int i = 0; i < n; i++){
							K_x. template block<3, 1>(idx, i) = res_temp_SO3 * (K_x. template block<3, 1>(idx, i));
						}
					}
					for(int i = 0; i < int(n); i++){
						L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
					int idx = (*it).first;
			
					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}
			
					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
		
					for(int i = 0; i < int(n); i++){
						L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
					}
					if(n > dof_Measurement)
					{
						for(int i = 0; i < int(dof_Measurement); i++){
							K_. template block<2, 1>(idx, i) = res_temp_S2 * (K_. template block<2, 1>(idx, i));
						}
					}
					else
					{
						for(int i = 0; i < n; i++){
							K_x. template block<2, 1>(idx, i) = res_temp_S2 * (K_x. template block<2, 1>(idx, i));
						}
					}
					for(int i = 0; i < int(n ); i++){
						L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
				if(n > dof_Measurement)
				{
					P_ = L_ - K_ * h_x_ * P_;
				}
				else
				{
					P_ = L_ - K_x * P_;
				}
				return;
			}
		}
	}

	template<typename measurement_dx>
	void update_iterated_dyn(measurement_dx h_x_new, int maximun_iter, std::vector<double> &limit, double ampl) {
		if(!(is_same<typename measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}
		int t = 0;
		dyn_cal vt_hz;
		vt_hz.v = true;
		vt_hz.t = true;
		state x_propagated = x_;
		cov P_propagated = P_;
		int dof_Measurement;
		for(int i=0; i<maximun_iter; i++)
		{
			vt_hz.v = true;
			MatrixXd R; 
			MatrixXd h_x_ = h_x_new(x_, R, vt_hz);
			VectorXd h_new_o = vt_hz.h_;
			VectorXd z_new_o = vt_hz.z_o;
			MatrixXd h_v_ = vt_hz.h_v;
			dof_Measurement = h_x_.rows();
			vectorized_state dx, dx_new;
			x_.boxminus(dx, x_propagated);
			dx_new = dx;
			if(! (vt_hz.v))
			{
				continue; // forever until the end of the current iteration
			}

			P_ = P_propagated;
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);
				}

				res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 3>(i, idx) =(P_. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}
		
			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}

				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			MatrixXd K_;
			if(n > dof_Measurement)
			{
				K_= P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_*R*h_v_.transpose()).inverse();
			}
			else
			{
				Eigen::MatrixXd R_in = (h_v_*R*h_v_.transpose()).inverse();
				K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in; 
			}
			cov K_x = K_ * h_x_;
			Matrix<scalar_type, n, 1> dx_ = K_ * (z_new_o - h_new_o) + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new; 
			state x_before = x_;
			x_.boxplus(dx_);
			vt_hz.t = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) * ampl > limit[i])
				{
					vt_hz.t = false;
					break;
				}
			}
			if(vt_hz.t) t++;
			if(t > 1 || i == maximun_iter - 1)
			{
				L_ = P_;
				std::cout << "iteration time:" << t << "," << i << std::endl;
		
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					for(int i = 0; i < int(n); i++){
						L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
					}
					if(n > dof_Measurement)
					{
						for(int i = 0; i < dof_Measurement; i++){
							K_. template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
						}
					}
					else
					{
						for(int i = 0; i < n; i++){
							K_x. template block<3, 1>(idx, i) = res_temp_SO3 * (K_x. template block<3, 1>(idx, i));
						}
					}
					for(int i = 0; i < int(n); i++){
						L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
					int idx = (*it).first;
			
					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}
			
					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
		
					for(int i = 0; i < int(n); i++){
						L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
					}
					if(n > dof_Measurement)
					{
						for(int i = 0; i < dof_Measurement; i++){
							K_. template block<2, 1>(idx, i) = res_temp_S2 * (K_. template block<2, 1>(idx, i));
						}
					}
					else
					{
						for(int i = 0; i < n; i++){
							K_x. template block<2, 1>(idx, i) = res_temp_S2 * (K_x. template block<2, 1>(idx, i));
						}
					}
					for(int i = 0; i < int(n ); i++){
						L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
				if(n > dof_Measurement)
				{
					P_ = L_ - K_*h_x_*P_;
				}
				else
				{
					P_ = L_ - K_x * P_;
				}
				return;
			}
		}
	}
	
	template<typename measurement_dx>
	void update_iterated_dyn_modified(measurement_dx h_x_new, double R, int maximun_iter, std::vector<double> &limit, double ampl) {
		if(!(is_same<typename measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}
		dyn_cal_modified vt_h;
		vt_h.v = true;
		vt_h.t = true;
		int t = 0;
		state x_propagated = x_;
		cov P_propagated = P_;
		int dof_Measurement; 
		vectorized_state dx_new = vectorized_state::Zero();
		for(int i=0; i<maximun_iter; i++)
		{
			vt_h.v = true;
			MatrixXd h_x_ = h_x_new(x_, vt_h);
			dof_Measurement = h_x_.rows();
			vectorized_state dx;
			x_.boxminus(dx, x_propagated);
			dx_new = dx;
			
			if(! vt_h.v)
			{
				continue; 
			}
			
			P_ = P_propagated;
			
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);
				}

				res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 3>(i, idx) =(P_. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}

			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}

				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			Eigen::Matrix<double, 12, 12> H_T_H = h_x_.transpose() * h_x_;
			MatrixXd K_;
			MatrixXd K_h_block;
			MatrixXd K_h = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, n);
			Eigen::Matrix<double, 23, 23> H = Eigen::Matrix<double, 23, 23>::Zero();
			H.block<3, 3>(0, 0) = H_T_H.block<3, 3>(0, 0);
			H.block<3, 3>(0, 6) = H_T_H.block<3, 3>(0, 3);
			H.block<3, 3>(0, 17) = H_T_H.block<3, 3>(0, 6);
			H.block<3, 3>(0, 20) = H_T_H.block<3, 3>(0, 9);
			H.block<3, 3>(6, 0) = H_T_H.block<3, 3>(3, 0);
			H.block<3, 3>(6, 6) = H_T_H.block<3, 3>(3, 3);
			H.block<3, 3>(6, 17) = H_T_H.block<3, 3>(3, 6);
			H.block<3, 3>(6, 20) = H_T_H.block<3, 3>(3, 9);
			H.block<3, 3>(17, 0) = H_T_H.block<3, 3>(6, 0);
			H.block<3, 3>(17, 6) = H_T_H.block<3, 3>(6, 3);
			H.block<3, 3>(17, 17) = H_T_H.block<3, 3>(6, 6);
			H.block<3, 3>(17, 20) = H_T_H.block<3, 3>(6, 9);
			H.block<3, 3>(20, 0) = H_T_H.block<3, 3>(9, 0);
			H.block<3, 3>(20, 6) = H_T_H.block<3, 3>(9, 3);
			H.block<3, 3>(20, 17) = H_T_H.block<3, 3>(9, 6);
			H.block<3, 3>(20, 20) = H_T_H.block<3, 3>(9, 9);
			if(n > dof_Measurement)
			{
				Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_ = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, 23);
				H_.col(0) = h_x_.col(0);
				H_.col(1) = h_x_.col(1);
				H_.col(2) = h_x_.col(2);
				H_.col(6) = h_x_.col(3);
				H_.col(7) = h_x_.col(4);
				H_.col(8) = h_x_.col(5);
				H_.col(17) = h_x_.col(6);
				H_.col(18) = h_x_.col(7);
				H_.col(19) = h_x_.col(8);
				H_.col(20) = h_x_.col(9);
				H_.col(21) = h_x_.col(10);
				H_.col(22) = h_x_.col(11);
				K_= P_ * H_.transpose() * (H_ * P_ * H_.transpose()/R + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse()/R;
			}
			else
			{
				cov K_i = (H + (P_/R).inverse()).inverse();
				Eigen::Matrix<double, 23, 12> K_b;
				K_b.col(0) = K_i.col(0);
				K_b.col(1) = K_i.col(1);
				K_b.col(2) = K_i.col(2);
				K_b.col(3) = K_i.col(6);
				K_b.col(4) = K_i.col(7);
				K_b.col(5) = K_i.col(8);
				K_b.col(6) = K_i.col(17);
				K_b.col(7) = K_i.col(18);
				K_b.col(8) = K_i.col(19);
				K_b.col(9) = K_i.col(20);
				K_b.col(10) = K_i.col(21);
				K_b.col(11) = K_i.col(22);
				K_ = K_b * h_x_.transpose();
				K_h_block = K_b * H_T_H;

				K_h.col(0) = K_h_block.col(0);
				K_h.col(1) = K_h_block.col(1);
				K_h.col(2) = K_h_block.col(2);
				K_h.col(6) = K_h_block.col(3);
				K_h.col(7) = K_h_block.col(4);
				K_h.col(8) = K_h_block.col(5);
				K_h.col(17) = K_h_block.col(6);
				K_h.col(18) = K_h_block.col(7);
				K_h.col(19) = K_h_block.col(8);
				K_h.col(20) = K_h_block.col(9);
				K_h.col(21) = K_h_block.col(10);
				K_h.col(22) = K_h_block.col(11);
			}
			Matrix<scalar_type, n, 1> dx_ = K_ * vt_h.h_ + (K_h - Matrix<scalar_type, n, n>::Identity()) * dx_new; 
			state x_before = x_;
			x_.boxplus(dx_);
			vt_h.t = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) * ampl > limit[i])
				{
					vt_h.t = false;
					break;
				}
			}
			if(vt_h.t) t++;

			if(t > 1 || i == maximun_iter - 1)
			{
				L_ = P_;
				std::cout << "iteration time" << t << "," << i << std::endl; 
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					for(int i = 0; i < int(n); i++){
						L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
					}
					for(int i = 0; i < n; i++){
						K_h. template block<3, 1>(idx, i) = res_temp_SO3 * (K_h. template block<3, 1>(idx, i));
					}
					for(int i = 0; i < int(n); i++){
						L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
					int idx = (*it).first;
			
					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}

					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
					for(int i = 0; i < int(n); i++){
						L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
					}
					for(int i = 0; i < n; i++){
						K_h. template block<2, 1>(idx, i) = res_temp_S2 * (K_h. template block<2, 1>(idx, i));
					}
					for(int i = 0; i < int(n ); i++){
						L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
				P_ = L_ - K_h * P_;
				return;
			}
		}
	}

	//template<typename measurementnoisecovariance>
	void update_sparse(measurement& z, measurementnoisecovariance &R) {
		
		if(!(is_same<typename measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}

		static const int dof_Measurement = measurement::DOF;

		spMt h_x_ = h_x(x_).sparseView();
		spMt h_v_ = h_v(x_).sparseView();
		Matrix<scalar_type, n, dof_Measurement> K_ = P_ * h_x_.transpose() * ((h_x_ * P_ * h_x_.transpose()) + h_v_ * R * h_v_.transpose()).inverse();
		Matrix<scalar_type, dof_Measurement, 1> innovation;
		z.boxminus(innovation, h(x_));
		Matrix<scalar_type, n, 1> dx = K_ * innovation;
        state x_before = x_;
		x_.boxplus(dx);

		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		L_ = P_;
		for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
			int idx = (*it).first;
			
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = dx(i + idx);
			}
			
			res_temp_SO3 = A_matrix(seg_SO3).transpose();
			for(int i = 0; i < int(n); i++){
				L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
			}
			for(int i = 0; i < int(dof_Measurement); i++){
				K_. template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
			}
			for(int i = 0; i < int(n); i++){
				L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
				P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
			}
		}


		
		Matrix<scalar_type, 2, 2> res_temp_S2;
		MTK::vect<2, scalar_type> seg_S2;
		for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			int idx = (*it).first;
			
			for(int i = 0; i < 2; i++){
				seg_S2(i) = dx(i + idx);
			}
			
			Matrix<scalar_type, 3, 2> expu_du;
			x_.S2_expu_u(expu_du, seg_S2, idx);
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			x_.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, seg_S2, idx);
			res_temp_S2 = Nx * Mx; 
			for(int i = 0; i < int(n); i++){
				L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
			}
			for(int i = 0; i < int(dof_Measurement); i++){
				K_. template block<2, 1>(idx, i) = res_temp_S2 * (K_. template block<2, 1>(idx, i));
			}
			for(int i = 0; i < int(n); i++){
				L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
			}
		}
		
		P_ = L_ - (K_ * h_x_ * P_); 
	}

	cov& change_P() {
		return P_;
	}

	state& change_x() {
		return x_;
	}

	void change_x(state &input_state)
	{
		x_ = input_state;
	}

	void change_P(cov &input_cov)
	{
		P_ = input_cov;
	}

	const state& get_x() const {
		return x_;
	}
	const cov& get_P() const {
		return P_;
	}
private:
	state x_;
	measurement m_;
	//input i_;
	cov P_;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	cov F_x1 = cov::Identity();
	cov F_x2 = cov::Identity();
	cov L_ = cov::Identity();
	/*
	auto *f;
	auto *f_x;
	auto *f_w;
	auto *h;
	auto *h_x;
	auto *h_v;
	*/
	processModel *f;
	processMatrix1 *f_x;
	processMatrix2 *f_w;

	measurementModel *h;
	measurementMatrix1 *h_x;
	measurementMatrix2 *h_v;
	
	template <typename T>
    T check_safe_update( T _temp_vec )
    {
        T temp_vec = _temp_vec;
        if ( std::isnan( temp_vec(0, 0) ) )
        {
            temp_vec.setZero();
            return temp_vec;
        }
        double angular_dis = temp_vec.block( 0, 0, 3, 1 ).norm() * 57.3;
        double pos_dis = temp_vec.block( 3, 0, 3, 1 ).norm();
        if ( angular_dis >= 20 || pos_dis > 1 )
        {
            printf( "Angular dis = %.2f, pos dis = %.2f\r\n", angular_dis, pos_dis );
            temp_vec.setZero();
        }
        return temp_vec;
    }
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<typename state, typename process_noise, typename measurement, typename measurement_noise, typename input = state>
class esekf{

	typedef esekf self;
	enum{
		n = state::DOF, m = state::DIM
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef Matrix<scalar_type, m, n> cov_;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;
	typedef Matrix<scalar_type, m, 1> flatted_state;
	typedef flatted_state processModel(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, process_noise::DOF> processMatrix2(state &, const input &);
	typedef typename MTK::get_cov<process_noise>::type processnoisecovariance;
	typedef measurement measurementModel(state &, bool &);
	typedef typename MTK::get_cross_cov<measurement, state>::type measurementMatrix1(state &, bool&);
	typedef typename MTK::get_cross_cov<measurement, measurement_noise>::type measurementMatrix2(state &, bool&);
	typedef typename MTK::get_cov<measurement_noise>::type measurementnoisecovariance;

	esekf(const state &x = state(),
		const cov  &P = cov::Identity()): x_(x), P_(P){
		SparseMatrix<scalar_type> ref(n, n);
		ref.setIdentity();
		l_ = ref;
		f_x_2 = ref;
		f_x_1 = ref;
	};

	void init(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel h_in, measurementMatrix1 h_x_in, measurementMatrix2 h_v_in)
	{
		f = f_in;
		f_x = f_x_in;
		f_w = f_w_in;
		h = h_in;
		h_x = h_x_in;
		h_v = h_v_in;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
	}

	void init(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in)
	{
		f = f_in;
		f_x = f_x_in;
		f_w = f_w_in;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
	}

	void predict(double &dt, processnoisecovariance &Q, const input &i_in = input()){
		flatted_state f_ = f(x_, i_in);
		cov_ f_x_ = f_x(x_, i_in);
		cov f_x_final;
		Matrix<scalar_type, m, process_noise::DOF> f_w_ = f_w(x_, i_in);
		Matrix<scalar_type, n, process_noise::DOF> f_w_final;
		state x_before = x_;
		x_.oplus(f_, dt);

		F_x1 = cov::Identity();
		for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
			int idx = (*it).first.first;
			int dim = (*it).first.second;
			int dof = (*it).second;
			for(int i = 0; i < int(n); i++){
				for(int j=0; j<dof; j++)
				{f_x_final(idx+j, i) = f_x_(dim+j, i);}	
			}
			for(int i = 0; i < int(f_w_.cols()); i++){
				for(int j=0; j<dof; j++)
				{f_w_final(idx+j, i) = f_w_(dim+j, i);}
			}
		}

		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = -1 * f_(dim + i) * dt;
			}

			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
			F_x1.template block<3, 3>(idx, idx) = res.toRotationMatrix();
			
			res_temp_SO3 = MTK::A_matrix(seg_SO3);
			for(int i = 0; i < int(n); i++){
				f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < int(f_w_.cols()); i++){
				f_w_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_. template block<3, 1>(dim, i));
			}
		}
		
		
		Matrix<scalar_type, 2, 3> res_temp_S2;
		MTK::vect<3, scalar_type> seg_S2;
		for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_S2(i) = f_(dim + i) * dt;
			}
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			x_.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, vec, idx);
			F_x1.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;

			Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
			x_before.S2_hat(x_before_hat, idx);
			res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat*MTK::A_matrix(seg_S2).transpose();
			
			for(int i = 0; i < int(n); i++){
				f_x_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_x_. template block<3, 1>(dim, i));
				
			}
			for(int i = 0; i < int(f_w_.cols() ); i++){
				f_w_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_w_. template block<3, 1>(dim, i));
			}
		}
		
		F_x1 += f_x_final * dt;
		P_ = (F_x1) * P_ * (F_x1).transpose() + (dt * f_w_final) * Q * (dt * f_w_final).transpose();
	}


	void predict_sparse(double &dt, processnoisecovariance &Q, const input &i_in = input()){
		flatted_state f_ = f(x_, i_in);
		state x_before = x_;
		x_.oplus(f_, dt);
		cov_ f_x_ = f_x(x_, i_in);
		cov f_x_final;
		Matrix<scalar_type, m, Eigen::Dynamic> f_w_ = f_w(x_, i_in);
		Matrix<scalar_type, n, process_noise::DOF> f_w_final;

		F_x1 = cov::Identity();
		for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
			int idx = (*it).first.first;
			int dim = (*it).first.second;
			int dof = (*it).second;
			for(int i = 0; i < int(n); i++){
				for(int j=0; j<dof; j++)
				{f_x_final(idx+j, i) = f_x_(dim+j, i);}	
			}
			for(int i = 0; i < int(f_w_.cols()); i++){
				for(int j=0; j<dof; j++)
				{f_w_final(idx+j, i) = f_w_(dim+j, i);}
			}
		}
		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) { 
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = -1 * f_(dim + i) * dt;
			}
			
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
			
			res_temp_SO3 = res.toRotationMatrix();
			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					f_x_1.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
				}
			}

			res_temp_SO3 = MTK::A_matrix(seg_SO3);
			for(int i = 0; i < int(n); i++){
				f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < int(f_w_.cols()); i++){
				f_w_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_. template block<3, 1>(dim, i));
			}
		}

		Matrix<scalar_type, 2, 3> res_temp_S2;
		Matrix<scalar_type, 2, 2> res_temp_S2_;
		MTK::vect<3, scalar_type> seg_S2;
		for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			int idx = (*it).first;
			int dim = (*it).second;
			for(int i = 0; i < 3; i++){
				seg_S2(i) = f_(dim + i) * dt;
			}
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			x_.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, vec, idx);
			res_temp_S2_ = Nx * res.toRotationMatrix() * Mx;

			Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
			x_before.S2_hat(x_before_hat, idx);
			res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat*MTK::A_matrix(seg_S2).transpose();

			for(int i = 0; i < 2; i++){
				for(int j = 0; j < 2; j++){
					f_x_1.coeffRef(idx + i, idx + j) = res_temp_S2_(i, j);
				}
			}

			for(int i = 0; i < int(n); i++){
				f_x_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_x_. template block<3, 1>(dim, i));	
			}
			for(int i = 0; i < int(f_w_.cols() ); i++){
				f_w_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_w_. template block<3, 1>(dim, i));
			}
		}

		f_x_1.makeCompressed();
		spMt f_x2 = f_x_final.sparseView();
		spMt f_w1 = f_w_final.sparseView();
		spMt xp = f_x_1 + f_x2 * dt;
		P_ = xp * P_ * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
	}


	void update_iterated(measurement& z, measurementnoisecovariance &R, int maximun_iter, std::vector<double> &limit, double ampl) {
		
		if(!(is_same<typename measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}
		int t = 0;
		bool converg = true;   
		bool valid = true;    
		state x_propagated = x_;
		cov P_propagated = P_;
		static const int dof_Measurement = measurement::DOF;
		for(int i=0; i<maximun_iter; i++)
		{
			vectorized_state dx, dx_new;
			x_.boxminus(dx, x_propagated);
			dx_new = dx;
			Matrix<scalar_type, dof_Measurement, n> h_x_ = h_x(x_, valid);
			Matrix<scalar_type, dof_Measurement, measurement_noise::DOF> h_v_ = h_v(x_, valid);
			if(! valid)
			{
				continue; 
			}

			P_ = P_propagated;
			
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);
				}

				res_temp_SO3 = A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx.template block<3, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_propagated. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 3>(i, idx) =(P_propagated. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}
		
		
			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}
				
				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx.template block<2, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<2, 1>(idx, i) = res_temp_S2 * (P_propagated. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 2>(i, idx) = (P_propagated. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			Matrix<scalar_type, n, dof_Measurement> K_;
			if(n > dof_Measurement)
			{
				K_= P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_ * R * h_v_.transpose()).inverse();
			}
			else
			{
				//Eigen::Matrix<scalar_type, measurement_noise::DOF, measurement_noise::DOF> R_in = R.inverse();
				measurementnoisecovariance R_in = (h_v_*R*h_v_.transpose()).inverse();
				K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in; // * R_in;
			}
			Matrix<scalar_type, dof_Measurement, 1> innovation; 
			z.boxminus(innovation, h(x_, valid));
			cov K_x = K_ * h_x_;
			Matrix<scalar_type, n, 1> dx_ = K_ * innovation + (K_x  - Matrix<scalar_type, n, n>::Identity()) * dx_new;
        	state x_before = x_;
			x_.boxplus(dx_);

			converg = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) * ampl > limit[i])
				{
					converg = false;
					break;
				}
			}

			if(converg) t++;
	        
			if(t > 1 || i == maximun_iter - 1)
			{
				L_ = P_;
		
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = A_matrix(seg_SO3).transpose();
					for(int i = 0; i < int(n); i++){
						L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
					}
					if(n > dof_Measurement)
					{
						for(int i = 0; i < int(dof_Measurement); i++){
							K_. template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
						}
					}
					else
					{
						for(int i = 0; i < n; i++){
							K_x. template block<3, 1>(idx, i) = res_temp_SO3 * (K_x. template block<3, 1>(idx, i));
						}
					}
					for(int i = 0; i < int(n); i++){
						L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
					int idx = (*it).first;
			
					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}
			
					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
		
					for(int i = 0; i < int(n); i++){
						L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
					}
					if(n > dof_Measurement)
					{
						for(int i = 0; i < int(dof_Measurement); i++){
							K_. template block<2, 1>(idx, i) = res_temp_S2 * (K_. template block<2, 1>(idx, i));
						}
					}
					else
					{
						for(int i = 0; i < n; i++){
							K_x. template block<2, 1>(idx, i) = res_temp_S2 * (K_x. template block<2, 1>(idx, i));
						}
					}
					for(int i = 0; i < int(n ); i++){
						L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
				if(n > dof_Measurement)
				{
					P_ = L_ - K_ * h_x_ * P_;
				}
				else
				{
					P_ = L_ - K_x * P_;
				}
				return;
			}
		}
	}

	template<typename measurement_dx>
	void update_iterated_dyn(measurement_dx h_x_new, int maximun_iter, std::vector<double> &limit, double ampl) {
		if(!(is_same<typename measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}
		int t = 0;
		dyn_cal vt_hz;
		vt_hz.v = true;
		vt_hz.t = true;
		state x_propagated = x_;
		cov P_propagated = P_;
		int dof_Measurement;
		for(int i=0; i<maximun_iter; i++)
		{
			vt_hz.v = true;
			MatrixXd R; 
			MatrixXd h_x_ = h_x_new(x_, R, vt_hz);
			VectorXd h_new_o = vt_hz.h_;
			VectorXd z_new_o = vt_hz.z_o;
			MatrixXd h_v_ = vt_hz.h_v;
			dof_Measurement = h_x_.rows();
			vectorized_state dx, dx_new;
			x_.boxminus(dx, x_propagated);
			dx_new = dx;
			if(! (vt_hz.v))
			{
				continue; // forever until the end of the current iteration
			}

			P_ = P_propagated;
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);
				}

				res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 3>(i, idx) =(P_. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}
		
			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}

				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			MatrixXd K_;
			if(n > dof_Measurement)
			{
				K_= P_ * h_x_.transpose() * (h_x_ * P_ * h_x_.transpose() + h_v_*R*h_v_.transpose()).inverse();
			}
			else
			{
				Eigen::MatrixXd R_in = (h_v_*R*h_v_.transpose()).inverse();
				K_ = (h_x_.transpose() * R_in * h_x_ + P_.inverse()).inverse() * h_x_.transpose() * R_in; 
			}
			cov K_x = K_ * h_x_;
			Matrix<scalar_type, n, 1> dx_ = K_ * (z_new_o - h_new_o) + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new; 
			state x_before = x_;
			x_.boxplus(dx_);
			vt_hz.t = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) * ampl > limit[i])
				{
					vt_hz.t = false;
					break;
				}
			}
			if(vt_hz.t) t++;
			if(t > 1 || i == maximun_iter - 1)
			{
				L_ = P_;
				std::cout << "iteration time:" << t << "," << i << std::endl;
		
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					for(int i = 0; i < int(n); i++){
						L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
					}
					if(n > dof_Measurement)
					{
						for(int i = 0; i < dof_Measurement; i++){
							K_. template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
						}
					}
					else
					{
						for(int i = 0; i < n; i++){
							K_x. template block<3, 1>(idx, i) = res_temp_SO3 * (K_x. template block<3, 1>(idx, i));
						}
					}
					for(int i = 0; i < int(n); i++){
						L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
					int idx = (*it).first;
			
					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}
			
					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
		
					for(int i = 0; i < int(n); i++){
						L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
					}
					if(n > dof_Measurement)
					{
						for(int i = 0; i < dof_Measurement; i++){
							K_. template block<2, 1>(idx, i) = res_temp_S2 * (K_. template block<2, 1>(idx, i));
						}
					}
					else
					{
						for(int i = 0; i < n; i++){
							K_x. template block<2, 1>(idx, i) = res_temp_S2 * (K_x. template block<2, 1>(idx, i));
						}
					}
					for(int i = 0; i < int(n ); i++){
						L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
				if(n > dof_Measurement)
				{
					P_ = L_ - K_*h_x_*P_;
				}
				else
				{
					P_ = L_ - K_x * P_;
				}
				return;
			}
		}
	}
	
	template<typename measurement_dx>
	void update_iterated_dyn_modified(measurement_dx h_x_new, double R, int maximun_iter, std::vector<double> &limit, double ampl) {
		if(!(is_same<typename measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}
		dyn_cal_modified vt_h;
		vt_h.v = true;
		vt_h.t = true;
		int t = 0;
		state x_propagated = x_;
		cov P_propagated = P_;
		int dof_Measurement; 
		vectorized_state dx_new = vectorized_state::Zero();
		for(int i=0; i<maximun_iter; i++)
		{
			vt_h.v = true;
			MatrixXd h_x_ = h_x_new(x_, vt_h);
			dof_Measurement = h_x_.rows();
			vectorized_state dx;
			x_.boxminus(dx, x_propagated);
			dx_new = dx;
			
			if(! vt_h.v)
			{
				continue; 
			}
			
			P_ = P_propagated;
			
			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = dx(idx+i);
				}

				res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
				dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 3>(i, idx) =(P_. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
				}
			}

			Matrix<scalar_type, 2, 2> res_temp_S2;
			MTK::vect<2, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 2; i++){
					seg_S2(i) = dx(idx + i);
				}

				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_.S2_Nx_yy(Nx, idx);
				x_propagated.S2_Mx(Mx, seg_S2, idx);
				res_temp_S2 = Nx * Mx; 
				dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
				for(int i = 0; i < int(n); i++){
					P_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i));	
				}
				for(int i = 0; i < int(n); i++){
					P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				}
			}

			Eigen::Matrix<double, 12, 12> H_T_H = h_x_.transpose() * h_x_;
			MatrixXd K_;
			MatrixXd K_h_block;
			MatrixXd K_h = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, n);
			Eigen::Matrix<double, 23, 23> H = Eigen::Matrix<double, 23, 23>::Zero();
			H.block<3, 3>(0, 0) = H_T_H.block<3, 3>(0, 0);
			H.block<3, 3>(0, 6) = H_T_H.block<3, 3>(0, 3);
			H.block<3, 3>(0, 17) = H_T_H.block<3, 3>(0, 6);
			H.block<3, 3>(0, 20) = H_T_H.block<3, 3>(0, 9);
			H.block<3, 3>(6, 0) = H_T_H.block<3, 3>(3, 0);
			H.block<3, 3>(6, 6) = H_T_H.block<3, 3>(3, 3);
			H.block<3, 3>(6, 17) = H_T_H.block<3, 3>(3, 6);
			H.block<3, 3>(6, 20) = H_T_H.block<3, 3>(3, 9);
			H.block<3, 3>(17, 0) = H_T_H.block<3, 3>(6, 0);
			H.block<3, 3>(17, 6) = H_T_H.block<3, 3>(6, 3);
			H.block<3, 3>(17, 17) = H_T_H.block<3, 3>(6, 6);
			H.block<3, 3>(17, 20) = H_T_H.block<3, 3>(6, 9);
			H.block<3, 3>(20, 0) = H_T_H.block<3, 3>(9, 0);
			H.block<3, 3>(20, 6) = H_T_H.block<3, 3>(9, 3);
			H.block<3, 3>(20, 17) = H_T_H.block<3, 3>(9, 6);
			H.block<3, 3>(20, 20) = H_T_H.block<3, 3>(9, 9);
			if(n > dof_Measurement)
			{
				Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_ = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, 23);
				H_.col(0) = h_x_.col(0);
				H_.col(1) = h_x_.col(1);
				H_.col(2) = h_x_.col(2);
				H_.col(6) = h_x_.col(3);
				H_.col(7) = h_x_.col(4);
				H_.col(8) = h_x_.col(5);
				H_.col(17) = h_x_.col(6);
				H_.col(18) = h_x_.col(7);
				H_.col(19) = h_x_.col(8);
				H_.col(20) = h_x_.col(9);
				H_.col(21) = h_x_.col(10);
				H_.col(22) = h_x_.col(11);
				K_= P_ * H_.transpose() * (H_ * P_ * H_.transpose()/R + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse()/R;
			}
			else
			{
				cov K_i = (H + (P_/R).inverse()).inverse();
				Eigen::Matrix<double, 23, 12> K_b;
				K_b.col(0) = K_i.col(0);
				K_b.col(1) = K_i.col(1);
				K_b.col(2) = K_i.col(2);
				K_b.col(3) = K_i.col(6);
				K_b.col(4) = K_i.col(7);
				K_b.col(5) = K_i.col(8);
				K_b.col(6) = K_i.col(17);
				K_b.col(7) = K_i.col(18);
				K_b.col(8) = K_i.col(19);
				K_b.col(9) = K_i.col(20);
				K_b.col(10) = K_i.col(21);
				K_b.col(11) = K_i.col(22);
				K_ = K_b * h_x_.transpose();
				K_h_block = K_b * H_T_H;

				K_h.col(0) = K_h_block.col(0);
				K_h.col(1) = K_h_block.col(1);
				K_h.col(2) = K_h_block.col(2);
				K_h.col(6) = K_h_block.col(3);
				K_h.col(7) = K_h_block.col(4);
				K_h.col(8) = K_h_block.col(5);
				K_h.col(17) = K_h_block.col(6);
				K_h.col(18) = K_h_block.col(7);
				K_h.col(19) = K_h_block.col(8);
				K_h.col(20) = K_h_block.col(9);
				K_h.col(21) = K_h_block.col(10);
				K_h.col(22) = K_h_block.col(11);
			}
			Matrix<scalar_type, n, 1> dx_ = K_ * vt_h.h_ + (K_h - Matrix<scalar_type, n, n>::Identity()) * dx_new; 
			state x_before = x_;
			x_.boxplus(dx_);
			vt_h.t = true;
			for(int i = 0; i < n ; i++)
			{
				if(std::fabs(dx_[i]) * ampl > limit[i])
				{
					vt_h.t = false;
					break;
				}
			}
			if(vt_h.t) t++;

			if(t > 1 || i == maximun_iter - 1)
			{
				L_ = P_;
				std::cout << "iteration time" << t << "," << i << std::endl; 
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
					int idx = (*it).first;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = dx_(i + idx);
					}
					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					for(int i = 0; i < int(n); i++){
						L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
					}
					for(int i = 0; i < n; i++){
						K_h. template block<3, 1>(idx, i) = res_temp_SO3 * (K_h. template block<3, 1>(idx, i));
					}
					for(int i = 0; i < int(n); i++){
						L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
					int idx = (*it).first;
			
					for(int i = 0; i < 2; i++){
						seg_S2(i) = dx_(i + idx);
					}

					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx; 
					for(int i = 0; i < int(n); i++){
						L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
					}
					for(int i = 0; i < n; i++){
						K_h. template block<2, 1>(idx, i) = res_temp_S2 * (K_h. template block<2, 1>(idx, i));
					}
					for(int i = 0; i < int(n ); i++){
						L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}
				P_ = L_ - K_h * P_;
				return;
			}
		}
	}

	void update_sparse(measurement& z, measurementnoisecovariance &R) {
		
		if(!(is_same<typename measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}

		static const int dof_Measurement = measurement::DOF;

		spMt h_x_ = h_x(x_).sparseView();
		spMt h_v_ = h_v(x_).sparseView();
		Matrix<scalar_type, n, dof_Measurement> K_ = P_ * h_x_.transpose() * ((h_x_ * P_ * h_x_.transpose()) + h_v_ * R * h_v_.transpose()).inverse();
		Matrix<scalar_type, dof_Measurement, 1> innovation;
		z.boxminus(innovation, h(x_));
		Matrix<scalar_type, n, 1> dx = K_ * innovation;
        state x_before = x_;
		x_.boxplus(dx);

		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		L_ = P_;
		for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
			int idx = (*it).first;
			
			for(int i = 0; i < 3; i++){
				seg_SO3(i) = dx(i + idx);
			}
			
			res_temp_SO3 = A_matrix(seg_SO3).transpose();
			for(int i = 0; i < int(n); i++){
				L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
			}
			for(int i = 0; i < int(dof_Measurement); i++){
				K_. template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
			}
			for(int i = 0; i < int(n); i++){
				L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
				P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
			}
		}


		
		Matrix<scalar_type, 2, 2> res_temp_S2;
		MTK::vect<2, scalar_type> seg_S2;
		for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			int idx = (*it).first;
			
			for(int i = 0; i < 2; i++){
				seg_S2(i) = dx(i + idx);
			}
			
			Matrix<scalar_type, 3, 2> expu_du;
			x_.S2_expu_u(expu_du, seg_S2, idx);
			Eigen::Matrix<scalar_type, 2, 3> Nx;
			Eigen::Matrix<scalar_type, 3, 2> Mx;
			x_.S2_Nx_yy(Nx, idx);
			x_before.S2_Mx(Mx, seg_S2, idx);
			res_temp_S2 = Nx * Mx; 
			for(int i = 0; i < int(n); i++){
				L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
			}
			for(int i = 0; i < int(dof_Measurement); i++){
				K_. template block<2, 1>(idx, i) = res_temp_S2 * (K_. template block<2, 1>(idx, i));
			}
			for(int i = 0; i < int(n); i++){
				L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
				P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
			}
		}
		
		P_ = L_ - (K_ * h_x_ * P_); 
	}

	cov& change_P() {
		return P_;
	}

	state& change_x() {
		return x_;
	}

	void change_x(state &input_state)
	{
		x_ = input_state;
	}

	void change_P(cov &input_cov)
	{
		P_ = input_cov;
	}

	const state& get_x() const {
		return x_;
	}
	const cov& get_P() const {
		return P_;
	}
private:
	state x_;
	measurement m_;
	input i_;
	cov P_;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	cov F_x1 = cov::Identity();
	cov F_x2 = cov::Identity();
	cov L_ = cov::Identity();

	processModel *f;
	processMatrix1 *f_x;
	processMatrix2 *f_w;

	measurementModel *h;
	measurementMatrix1 *h_x;
	measurementMatrix2 *h_v;

	template <typename T>
    T check_safe_update( T _temp_vec )
    {
        T temp_vec = _temp_vec;
        if ( std::isnan( temp_vec(0, 0) ) )
        {
            temp_vec.setZero();
            return temp_vec;
        }
        double angular_dis = temp_vec.block( 0, 0, 3, 1 ).norm() * 57.3;
        double pos_dis = temp_vec.block( 3, 0, 3, 1 ).norm();
        if ( angular_dis >= 20 || pos_dis > 1 )
        {
            printf( "Angular dis = %.2f, pos dis = %.2f\r\n", angular_dis, pos_dis );
            temp_vec.setZero();
        }
        return temp_vec;
    }
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
