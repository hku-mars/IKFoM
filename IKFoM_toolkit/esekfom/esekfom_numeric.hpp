#ifndef ESEKFOM_EKF_NUMERIC_HPP
#define ESEKFOM_EKF_NUMERIC_HPP


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

template<typename state>
class esekf_numeric{

	typedef esekf_numeric self;
	enum {
		n = state::DOF
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;

	esekf_numeric(const state &x = state(),
		const cov &P = cov::Identity()): x_(x), P_(P){
		
		SparseMatrix<scalar_type> ref(n, n);
		ref.setIdentity();
		l_ = ref;
		f_x_2 = ref;
		f_x_1 = ref;
	};

	template<typename processModel, typename processnoise, typename processnoisecovariance> 
	void predict(processModel f, processnoise &w_i, processnoisecovariance &Q, double dt, std::vector<int> &group_of_SO3, std::vector<int> &group_of_S2){
		static const int dof_processnoise = processnoise::DOF;
		processnoise w;
		vectorized_state f_ = f(x_, w); 
		state x_before = x_;
		x_.boxplus(f_, dt);

		double min_vary = MTK::tolerance<scalar_type>();

		cov f_x;
		vectorized_state u = vectorized_state::Zero();
		state x_temp;
		for (int i = 0; i < n; i++) {
			u[i] = min_vary;
			x_temp = x_;
			x_temp.boxplus(u);
			f_x.col(i) = (f(x_temp, w) - f(x_, w)) / min_vary;
			u[i] = 0;
		}

		Matrix<scalar_type, n, dof_processnoise> f_w;
		processnoise w0;
		Matrix<scalar_type, dof_processnoise, 1> w0_ = Matrix<scalar_type, dof_processnoise, 1>::Zero();
		for (int i = 0; i < dof_processnoise; i++) {
			w0 = w;
			w0_[i] = min_vary;
			w0.boxplus(w0_);
			f_w.col(i) = (f(x_, w0) - f(x_, w)) / min_vary;
			w0_[i] = 0;
		}

		F_x1 = cov::Identity();

		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<int>::iterator it = group_of_SO3.begin(); it != group_of_SO3.end(); it++) {
			int idx = *it;
			
			for (int i = 0; i < 3; i++) {
				seg_SO3(i) = -1 * f_(idx + i) * dt;
			}
			
			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
			F_x1.template block<3, 3>(idx, idx) = res.toRotationMatrix();
			
			res_temp_SO3 = A_matrix(seg_SO3);
			for (int i = 0; i < int(n / 3); i++) {
				f_x. template block<3, 3>(idx, i * 3) = res_temp_SO3 * (f_x. template block<3, 3>(idx, i * 3));

			}
			for (int i = 0; i < int(dof_processnoise / 3); i++) {
				f_w. template block<3, 3>(idx, i * 3) = res_temp_SO3 * (f_w. template block<3, 3>(idx, i * 3));
			}
		}

		Matrix<scalar_type, 2, 2> res_temp_S2;
		MTK::vect<2, scalar_type> seg_S2;
		for (std::vector<int>::iterator it = group_of_S2.begin(); it != group_of_S2.end(); it++) 
		{
			int idx = *(it);
			
			for(int i = 0; i < 2; i++){
				seg_S2(i) = f_(idx + i) * dt;
			}
/*
			Matrix<scalar_type, 2, 3> w_expw = Matrix<scalar_type, 2, 3>::Zero();
			w_expw(0, 1) = 1;
			w_expw(1, 2) = 1;
*/
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			Matrix<scalar_type, 2, 3> w_expw_ = x_.w_expw(vec, idx);
			Matrix<scalar_type, 3, 2> expu_u_ = x_.expu_u(vec, idx);
			
			Matrix<scalar_type, 3, 3> x_after_Rx = Matrix<scalar_type, 3, 3>::Zero();
			x_.S2_Rx(x_after_Rx, idx);
			Matrix<scalar_type, 3, 3> x_before_Rx = Matrix<scalar_type, 3, 3>::Zero();
			x_before.S2_Rx(x_before_Rx, idx);
			Matrix<scalar_type, 3, 3> xplusu_dx = Matrix<scalar_type, 3, 3>::Zero();
			x_before.S2_xplusu_dx(xplusu_dx, seg_S2, idx);
			F_x1.template block<2, 2>(idx, idx) = w_expw_ * x_after_Rx.transpose() * xplusu_dx * x_before_Rx * expu_u_;
			Matrix<scalar_type, 3, 2> expu_du = x_.expu_u(seg_S2, idx);
			
			res_temp_S2 = w_expw_ * x_after_Rx.transpose() * x_before_Rx * expu_du;
			
			for(int i = 0; i < int(n / 2); i++){
				f_x. template block<2, 2>(idx, i * 2) = res_temp_S2 * (f_x. template block<2, 2>(idx, i * 2));
				
			}
			for(int i = 0; i < int(dof_processnoise / 2); i++){
				f_w. template block<2, 2>(idx, i * 2) = res_temp_S2 * (f_w. template block<2, 2>(idx, i * 2));
			}
		}

		F_x1 += f_x * dt;
		P_ = (F_x1) * P_ * (F_x1).transpose() + (dt * f_w) * Q * (f_w * dt).transpose();

	}


	template<typename processModel, typename processnoise, typename processnoisecovariance>
	void predict_sparse(processModel f, processnoise &w_i, processnoisecovariance &Q, double dt, std::vector<int>& group_of_SO3, std::vector<int> &group_of_S2) {
		static const int dof_processnoise = processnoise::DOF;
		processnoise w;
		vectorized_state f_ = f(x_, w); 
		state x_before = x_;
		x_.boxplus(f_, dt);

		double min_vary = MTK::tolerance<scalar_type>();

		cov f_x;
		vectorized_state u = vectorized_state::Zero();
		state x_temp;
		for (int i = 0; i < n; i++) {
			u[i] = min_vary;
			x_temp = x_;
			x_temp.boxplus(u);
			f_x.col(i) = (f(x_temp, w) - f(x_, w)) / min_vary;
			u[i] = 0;
		}

		Matrix<scalar_type, n, dof_processnoise> f_w;
		processnoise w0;
		Matrix<scalar_type, dof_processnoise, 1> w0_ = Matrix<scalar_type, dof_processnoise, 1>::Zero();
		for (int i = 0; i < dof_processnoise; i++) {
			w0 = w;
			w0_[i] = min_vary;
			w0.boxplus(w0_);
			f_w.col(i) = (f(x_, w0) - f(x_, w)) / min_vary;
			w0_[i] = 0;
		}
		
		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (std::vector<int>::iterator it = group_of_SO3.begin(); it != group_of_SO3.end(); it++) { 
			int idx = *it;
			
			for (int i = 0; i < 3; i++) {
				seg_SO3(i) = -1 * f_(idx + i) * dt;
			}

			MTK::SO3<scalar_type> res;
			res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1 / 2));
			res_temp_SO3 = res.toRotationMatrix();
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					f_x_1.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
				}
			}

			res_temp_SO3 = A_matrix(seg_SO3);

			for (int i = 0; i < int(n / 3); i++) {
				f_x. template block<3, 3>(idx, i * 3) = res_temp_SO3 * (f_x. template block<3, 3>(idx, i * 3));

			}
			for (int i = 0; i < int(dof_processnoise / 3); i++) {
				f_w. template block<3, 3>(idx, i * 3) = res_temp_SO3 * (f_w. template block<3, 3>(idx, i * 3));
			}
		}

		Matrix<scalar_type, 2, 2> res_temp_S2;
		MTK::vect<2, scalar_type> seg_S2;
		for (std::vector<int>::iterator it = group_of_S2.begin(); it != group_of_S2.end(); it++) {
			int idx = *(it);
			
			for(int i = 0; i < 2; i++){
				seg_S2(i) = f_(idx + i) * dt;
			}
/*
			Matrix<scalar_type, 2, 3> w_expw = Matrix<scalar_type, 2, 3>::Zero();
			w_expw(0, 1) = 1;
			w_expw(1, 2) = 1;
*/
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			Matrix<scalar_type, 2, 3> w_expw_ = x_.w_expw(vec, idx);
			Matrix<scalar_type, 3, 2> expu_u_ = x_.expu_u(vec, idx);
						
			Matrix<scalar_type, 3, 2> expu_du = x_.expu_u(seg_S2, idx);
			Matrix<scalar_type, 3, 3> x_after_Rx;
			x_.S2_Rx(x_after_Rx, idx);
			Matrix<scalar_type, 3, 3> x_before_Rx = Matrix<scalar_type, 3, 3>::Zero();
			x_before.S2_Rx(x_before_Rx, idx);
			Matrix<scalar_type, 3, 3> xplusu_dx = Matrix<scalar_type, 3, 3>::Zero();
			x_before.S2_xplusu_dx(xplusu_dx, seg_S2, idx);

			res_temp_S2 = w_expw_ * x_after_Rx.transpose() * xplusu_dx * x_before_Rx * expu_u_;
			for(int i = 0; i < 2; i++){
				for(int j = 0; j < 2; j++){
					f_x_1.coeffRef(idx + i, idx + j) = res_temp_S2(i, j);
				}
			}
			
			res_temp_S2 = w_expw_ * x_after_Rx.transpose() * x_before_Rx * expu_du;
			for(int i = 0; i < int(n / 2); i++){
				f_x. template block<2, 2>(idx, i * 2) = res_temp_S2 * (f_x. template block<2, 2>(idx, i * 2));	
			}
			for(int i = 0; i < int(dof_processnoise / 2); i++){
				f_w. template block<2, 2>(idx, i * 2) = res_temp_S2 * (f_w. template block<2, 2>(idx, i * 2));
			}
		}

		f_x_1.makeCompressed();
		spMt f_x2 = f_x.sparseView(); 
		spMt f_w1 = f_w.sparseView();
		spMt xp = f_x_1 + f_x2 * dt;
		P_ = xp * P_ * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
	}


	template<typename Measurement, typename measurementModel, typename measurementnoise, typename measurementnoisecovariance>
	void update(Measurement& z, measurementModel h, measurementnoise &v_i, measurementnoisecovariance &R, std::vector<int> &group_of_SO3, std::vector<int> &group_of_S2) {
		
		if(!(is_same<typename Measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}

		static const int dof_measurementnoise = measurementnoise::DOF;
		static const int dof_Measurement = Measurement::DOF;
		typedef Matrix<scalar_type, dof_Measurement, 1> vectorized_measurement;
		
		double min_vary = MTK::tolerance<scalar_type>();
		measurementnoise v;
		vectorized_state u = vectorized_state::Zero();
		Matrix<scalar_type, dof_Measurement, n> z_x;
		state x_temp;
		vectorized_measurement dh;
		for(int i = 0; i < n; i++) {
			u[i] = min_vary;
			x_temp = x_;
			x_temp.boxplus(u);
			h(x_temp, v).boxminus(dh, h(x_, v));
			z_x.col(i) = dh / min_vary;
			u[i] = 0;
		}

		Matrix<scalar_type, dof_Measurement, dof_measurementnoise> z_v;
		measurementnoise v0;
		Matrix<scalar_type, dof_measurementnoise, 1> v0_ = Matrix<scalar_type, dof_measurementnoise, 1>::Zero();
		for (int i = 0; i < dof_measurementnoise; i++) {
			v0 = v;
			v0_[i] = min_vary;
			v0.boxplus(v0_);
			h(x_, v0).boxminus(dh, h(x_, v));
			z_v.col(i) = dh / min_vary;
			v0_[i] = 0;
		}
			
		vectorized_measurement innovation; 
		z.boxminus(innovation, h(x_, v));
		Matrix<scalar_type, n, dof_Measurement> K_ = P_ * z_x.transpose() * (z_x * P_ * z_x.transpose() + z_v * R * z_v.transpose()).inverse();
		vectorized_state dx = K_ * innovation;
        state x_before = x_;
		x_.boxplus(dx);

		L_ = P_;

		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (typename std::vector<int>::iterator it = group_of_SO3.begin(); it != group_of_SO3.end(); it++) {
			int idx = *it;
			
			for (int i = 0; i < 3; i++) {
				seg_SO3(i) = dx(i + idx);
			}
			
			res_temp_SO3 = A_matrix(seg_SO3).transpose();
			for (int i = 0; i < int(n / 3); i++) {
				L_. template block<3, 3>(idx, i * 3) = res_temp_SO3 * (P_. template block<3, 3>(idx, i * 3));
			}
			for (int i = 0; i < int(dof_Measurement / 3); i++) {
				K_. template block<3, 3>(idx, i * 3) = res_temp_SO3 * (K_. template block<3, 3>(idx, i * 3));
			}
			for (int i = 0; i < int(n / 3); i++) {
				L_. template block<3, 3>(i * 3, idx) = (L_. template block<3, 3>(i * 3, idx)) * res_temp_SO3.transpose();
				P_. template block<3, 3>(i * 3, idx) = (P_. template block<3, 3>(i * 3, idx)) * res_temp_SO3.transpose();
			}
		}

		Matrix<scalar_type, 2, 2> res_temp_S2;
		MTK::vect<2, scalar_type> seg_S2;
		for(typename std::vector<int>::iterator it = group_of_S2.begin(); it != group_of_S2.end(); it++) {
			int idx = *it;
			
			for(int i = 0; i < 2; i++){
				seg_S2(i) = dx(i + idx);
			}

			Matrix<scalar_type, 3, 2> expu_du = x_.expu_u(seg_S2, idx);
			/*
			Matrix<scalar_type, 2, 3> w_expw = Matrix<scalar_type, 2, 3>::Zero();
			w_expw(0, 1) = 1;
			w_expw(1, 2) = 1;
			*/
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			Matrix<scalar_type, 2, 3> w_expw_ = x_.w_expw(vec, idx);
			
			Matrix<scalar_type, 3, 3> x_after_Rx = Matrix<scalar_type, 3, 3>::Zero();
			x_.S2_Rx(x_after_Rx, idx);
			Matrix<scalar_type, 3, 3> x_before_Rx = Matrix<scalar_type, 3, 3>::Zero();
			x_before.S2_Rx(x_before_Rx, idx);

			res_temp_S2 = w_expw_ * x_after_Rx.transpose() * x_before_Rx * expu_du;
			for(int i = 0; i < int(n / 2); i++){
				L_. template block<2, 2>(idx, i * 2) = res_temp_S2 * (P_. template block<2, 2>(idx, i * 2)); 
			}
			for(int i = 0; i < int(dof_Measurement / 2); i++){
				K_. template block<2, 2>(idx, i * 2) = res_temp_S2 * (K_. template block<2, 2>(idx, i * 2));
			}
			for(int i = 0; i < int(n / 2); i++){
				L_. template block<2, 2>(i * 2, idx) = (L_. template block<2, 2>(i * 2, idx)) * res_temp_S2.transpose();
				P_. template block<2, 2>(i * 2, idx) = (P_. template block<2, 2>(i * 2, idx)) * res_temp_S2.transpose();
			}
		}

		P_ = L_ - K_ * z_x * P_;
	}


	template<typename Measurement, typename measurementModel, typename measurementnoise, typename measurementnoisecovariance>
	void update_sparse(Measurement& z, measurementModel h, measurementnoise &v_i, measurementnoisecovariance &R, std::vector<int>& group_of_SO3, std::vector<int> &group_of_S2) {
	
		if(!(is_same<typename Measurement::scalar, scalar_type>())){
			std::cerr << "the scalar type of measurment must be the same as the state" << std::endl;
			std::exit(100);
		}

		static const int dof_measurementnoise = measurementnoise::DOF;
		static const int dof_Measurement = Measurement::DOF;
		typedef Matrix<scalar_type, dof_Measurement, 1> vectorized_measurement;
		
		double min_vary = MTK::tolerance<scalar_type>();
		measurementnoise v;
		vectorized_state u = vectorized_state::Zero();
		Matrix<scalar_type, dof_Measurement, n> z_x;
		state x_temp;
		vectorized_measurement dh;
		for (int i = 0; i < n; i++) {
			u[i] = min_vary;
			x_temp = x_;
			x_temp.boxplus(u);
			h(x_temp, v).boxminus(dh, h(x_, v));
			z_x.col(i) = dh / min_vary;
			u[i] = 0;
		}

		Matrix<scalar_type, dof_Measurement, dof_measurementnoise> z_v;
		measurementnoise v0;
		Matrix<scalar_type, dof_measurementnoise, 1> v0_ = Matrix<scalar_type, dof_measurementnoise, 1>::Zero();
		for (int i = 0; i < dof_measurementnoise; i++) {
			v0 = v;
			v0_[i] = min_vary;
			v0.boxplus(v0_);
			h(x_, v0).boxminus(dh, h(x_, v));
			z_v.col(i) = dh / min_vary;
			v0_[i] = 0;
		}

		spMt z_x_ = z_x.sparseView();
		spMt z_v_ = z_v.sparseView();
		
		Matrix<scalar_type, n, dof_Measurement> K_ = P_ * z_x_.transpose() * ((z_x_ * P_ * z_x_.transpose()) + (z_v_) * R * (z_v_.transpose())).inverse();
		Matrix<scalar_type, dof_Measurement, 1> innovation; 
		z.boxminus(innovation, h(x_, v));
		Matrix<scalar_type, n, 1> dx = K_ * innovation;
        state x_before = x_;
		x_.boxplus(dx);

		L_ = P_;

		Matrix<scalar_type, 3, 3> res_temp_SO3;
		MTK::vect<3, scalar_type> seg_SO3;
		for (typename std::vector<int>::iterator it = group_of_SO3.begin(); it != group_of_SO3.end(); it++) {
			int idx = *it;
			
			for (int i = 0; i < 3; i++) {
				seg_SO3(i) = dx(i + idx);
			}
			
			res_temp_SO3 = A_matrix(seg_SO3).transpose();
			for (int i = 0; i < int(n / 3); i++) {
				L_. template block<3, 3>(idx, i * 3) = res_temp_SO3 * (P_. template block<3, 3>(idx, i * 3));
			}
			for (int i = 0; i < int(dof_Measurement / 3); i++) {
				K_. template block<3, 3>(idx, i * 3) = res_temp_SO3 * (K_. template block<3, 3>(idx, i * 3));
			}
			for (int i = 0; i < int(n / 3); i++) {
				L_. template block<3, 3>(i * 3, idx) = (L_. template block<3, 3>(i * 3, idx)) * res_temp_SO3.transpose();
				P_. template block<3, 3>(i * 3, idx) = (P_. template block<3, 3>(i * 3, idx)) * res_temp_SO3.transpose();
			}
		}

		
		Matrix<scalar_type, 2, 2> res_temp_S2;
		MTK::vect<2, scalar_type> seg_S2;
		for(typename std::vector<int>::iterator it = group_of_S2.begin(); it != group_of_S2.end(); it++) {
			int idx = *it;
			
			for(int i = 0; i < 2; i++){
				seg_S2(i) = dx(i + idx);
			}
			
			Matrix<scalar_type, 3, 2> expu_du = x_.expu_u(seg_S2, idx);
/*
			Matrix<scalar_type, 2, 3> w_expw = Matrix<scalar_type, 2, 3>::Zero();
			w_expw(0, 1) = 1;
			w_expw(1, 2) = 1;
*/
			MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
			Matrix<scalar_type, 2, 3> w_expw_ = x_.w_expw(vec, idx);
						
			Matrix<scalar_type, 3, 3> x_after_Rx = Matrix<scalar_type, 3, 3>::Zero();
			x_.S2_Rx(x_after_Rx, idx);
			Matrix<scalar_type, 3, 3> x_before_Rx = Matrix<scalar_type, 3, 3>::Zero();
			x_before.S2_Rx(x_before_Rx, idx);

			res_temp_S2 = w_expw_ * x_after_Rx.transpose() * x_before_Rx * expu_du;
			for(int i = 0; i < int(n / 2); i++){
				L_. template block<2, 2>(idx, i * 2) = res_temp_S2 * (P_. template block<2, 2>(idx, i * 2)); 
			}
			for(int i = 0; i < int(dof_Measurement / 2); i++){
				K_. template block<2, 2>(idx, i * 2) = res_temp_S2 * (K_. template block<2, 2>(idx, i * 2));
			}
			for(int i = 0; i < int(n / 2); i++){
				L_. template block<2, 2>(i * 2, idx) = (L_. template block<2, 2>(i * 2, idx)) * res_temp_S2.transpose();
				P_. template block<2, 2>(i * 2, idx) = (P_. template block<2, 2>(i * 2, idx)) * res_temp_S2.transpose();
			}
		}

		P_ = L_ - (K_ * z_x_ * P_); 
	}

	cov& change_P() {
		return P_;
	}

	state& change_x() {
		return x_;
	}

	const state& get_x() const {
		return x_;
	}
	const cov& get_P() const {
		return P_;
	}
private:
	state x_;
	cov P_;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	cov F_x1 = cov::Identity();
	cov F_x2 = cov::Identity();
	cov L_ = cov::Identity();

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

#endif // ESEKFOM_EKF_NUMERIC_HPP
