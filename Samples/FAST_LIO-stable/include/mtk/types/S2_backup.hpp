/*
 *  Copyright (c) 2008--2011, Universitaet Bremen
 *  All rights reserved.
 *
 *  Author: Christoph Hertzberg <chtz@informatik.uni-bremen.de>
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
/**
 * @file mtk/types/S2.hpp
 * @brief Unit vectors on the sphere, or directions in 3D.
 */
#ifndef S2_H_
#define S2_H_


#include "vect.hpp"

#include "SOn.hpp"
#include "../src/mtkmath.hpp"




namespace MTK {

/**
 * Manifold representation of @f$ S^2 @f$. 
 * Used for unit vectors on the sphere or directions in 3D.
 * 
 * @todo add conversions from/to polar angles?
 */
template<class _scalar = double>
struct S2 {
	
	typedef _scalar scalar;
	typedef vect<3, scalar> vect_type; 
	typedef typename vect_type::base vec3; 
	enum {DOF=2};
	
//private:
	/**
	 * Unit vector on the sphere, or vector pointing in a direction
	 */
	vect_type vec; 
	
public:
	S2() : vec(vec3(0, 0, std::sqrt(1))) { } 
	S2(const scalar &x, const scalar &y, const scalar &z) : vec(vec3(x, y, z)) { 
		vec.normalize();
	}
	
	S2(const vect_type &_vec) : vec(_vec) {
		vec.normalize();
	}
	
	void boxplus(MTK::vectview<const scalar, 2> delta, scalar scale=1) {
		
		vect_type exp_delta;
		exp_delta[0] = MTK::exp(MTK::vectview<scalar, 2>(exp_delta.template tail<2>()), delta, scale); 
		vec = rotate(exp_delta, true); 
	} 
	
	void boxminus(MTK::vectview<scalar, 2> res, const S2<scalar>& other) const {
		const vect_type rotated = other.rotate(vec, false);
		
		MTK::log(res, rotated[0], rotated.template tail<2>(), scalar(1.0), false);
	}
	
	void S2_Rx(Eigen::Matrix<scalar, 3, 3> &res){
		
		if(vec[2] + 1.0 > 1e-11)
		{
			
			res << vec[0], vec[0]*vec[1]/(1+vec[2]), vec[2]+vec[1]*vec[1]/(1+vec[2]),
					vec[1], -vec[2]-vec[0]*vec[0]/(1+vec[2]), -vec[0]*vec[1]/(1+vec[2]),
					vec[2], vec[1], -vec[0];
		}
		else
		{
			res = Eigen::Matrix<scalar, 3, 3>::Zero();
			res(0, 2) = -1;
			res(1, 1) = -1;
			res(2, 0) = 1;
		}
	}

	void S2_xplusu_dx(Eigen::Matrix<scalar, 3, 3> &res, MTK::vectview<const scalar, 2> delta, scalar scale=1){
		vect_type exp_delta;
		exp_delta[0] = MTK::exp(MTK::vectview<scalar, 2>(exp_delta.template tail<2>()), delta, scale);
		if(vec[2] + 1.0 > 1e-11)
		{
			res << exp_delta[0]+vec[1]/(1+vec[2])*exp_delta[1], vec[0]/(1+vec[2])*exp_delta[1]+2*vec[1]/(1+vec[2])*exp_delta[2], -vec[0]*vec[1]/(1+vec[2])/(1+vec[2])*exp_delta[1]+exp_delta[2]-vec[1]*vec[1]/(1+vec[2])/(1+vec[2])*exp_delta[2],
					-2*vec[0]/(1+vec[2])*exp_delta[1]-vec[1]/(1+vec[2])*exp_delta[2], exp_delta[0]-vec[0]/(1+vec[2])*exp_delta[2], -exp_delta[1]+vec[0]*vec[0]/(1+vec[2])/(1+vec[2])*exp_delta[1]+vec[0]*vec[1]/(1+vec[2])/(1+vec[2])*exp_delta[2],
					-exp_delta[2], exp_delta[1], exp_delta[0];
		}
		else
		{
			if(exp_delta[0] + 1.0 > 1e-11)
			{
				res << 1-exp_delta[2]*exp_delta[2]/(1+exp_delta[0]), -exp_delta[1]*exp_delta[2]/(1+exp_delta[0]), exp_delta[2],
						-exp_delta[1]*exp_delta[2]/(1+exp_delta[0]), 1-exp_delta[1]*exp_delta[1]/(1+exp_delta[0]), exp_delta[1],
						-exp_delta[2], -exp_delta[1], 1;
			}
			else
			{
				res = -Eigen::Matrix<scalar, 3, 3>::Identity();
			}
			
			/*
			res = Eigen::Matrix<scalar, 3, 3>::Zero();
			res(0, 2) = -1;
			res(1, 1) = -1;
			res(2, 0) = 1;
			*/
		}
		
		/*
		double alpha = std::atan2(vec[2], vec[1]);
		double r = std::sqrt(vec[2]*vec[2] + vec[1]*vec[1]);
		double v = std::sqrt(delta[0]*delta[0]+delta[1]*delta[1]);
		if(r < 1e-11){
			//r = 1e-11;
			
			if(v < 1e-11){
				res = Eigen::Matrix<scalar, 3, 3>::Identity();
			}
			else{
				if(vec[0] > 0){
					Eigen::Matrix<scalar, 3, 3> hat, hat_2;
					hat << 0, -delta[1], delta[0],
						delta[1], 0, 0,
						-delta[0], 0, 0;
					hat_2 << -delta[0]*delta[0] - delta[1]*delta[1], 0, 0,
						0, -delta[1]*delta[1], delta[1]*delta[0],
						0, delta[0]*delta[1], -delta[0]*delta[0];
					res = Eigen::Matrix<scalar, 3, 3>::Identity() + std::sin(v)/v*hat + (1 - std::cos(v))/v/v*hat_2;
				}
				else{
					Eigen::Matrix<scalar, 3, 3> hat, hat_2;
					hat << 0, -delta[1], -delta[0],
						delta[1], 0, 0,
						delta[0], 0, 0;
					hat_2 << -delta[0]*delta[0] - delta[1]*delta[1], 0, 0,
						0, -delta[1]*delta[1], -delta[1]*delta[0],
						0, -delta[0]*delta[1], -delta[0]*delta[0];
					res = Eigen::Matrix<scalar, 3, 3>::Identity() + std::sin(v)/v*hat + (1 - std::cos(v))/v/v*hat_2;
				}
			}
				
			
		}
		else{
			double c = cos(alpha), s = sin(alpha);

			vect_type exp_delta;
			exp_delta[0] = MTK::exp(MTK::vectview<scalar, 2>(exp_delta.template tail<2>()), delta, scale);
		//std::cout << exp_delta << std::endl;
			res << exp_delta[0], exp_delta[1]*c, exp_delta[1]*s,
					c*exp_delta[1], exp_delta[0]+vec[0]*exp_delta[1]*s*s/r+exp_delta[2]*s*c/r, -vec[0]*exp_delta[1]*c*s/r-exp_delta[2]*c*c/r,
					s*exp_delta[1], -vec[0]*exp_delta[1]*c*s/r+exp_delta[2]*s*s/r, exp_delta[0]+vec[0]*exp_delta[1]*c*c/r-exp_delta[2]*s*c/r;
		}
		*/
	}

	operator const vect_type&() const{//operatorд���˷���ֵ����֮ǰ��
		return vec;
	}// vec�Ǹ�ʵ�����Ķ���
	
	const vect_type& get_vect() const {
		return vec;
	}
	
	friend S2 operator*(const SO3<scalar>& rot, const S2& dir)
	{
		S2 ret;
		ret.vec = rot * dir.vec;
		return ret;
	}
	
	scalar operator[](int idx) const {return vec[idx]; }
	
	friend std::ostream& operator<<(std::ostream &os, const S2<scalar>& vec){
		return os << vec.vec.transpose() << " ";
	}//���������
	friend std::istream& operator>>(std::istream &is, S2<scalar>& vec){
		for(int i=0; i<3; ++i)
			is >> vec.vec[i];
		vec.vec.normalize();
		return is;
	}
	
private:
	/**
	 * For inverse=false rotates oth, such that oth==*this implies result is [1 0 0],
	 * for inverse=true rotates such that @c oth ==[1 0 0] implies result equals *this.
	 * Function is smooth w.r.t. @c oth, but not necessarily continuous w.r.t. @c *this 
	 */
	vect_type rotate(const vect_type& oth, bool inverse) const {//��box����������õ��ˡ�
		vect_type ret;
		Eigen::Matrix<scalar, 3, 3> Rx;
		if(vec[2] + 1.0 > 1e-11)
		{
			Rx << vec[0], vec[0]*vec[1]/(1+vec[2]), vec[2]+vec[1]*vec[1]/(1+vec[2]),
					vec[1], -vec[2]-vec[0]*vec[0]/(1+vec[2]), -vec[0]*vec[1]/(1+vec[2]),
					vec[2], vec[1], -vec[0];
			
		}
		else
		{
			Rx = Eigen::Matrix<scalar, 3, 3>::Zero();
			Rx(0, 2) = -1;
			Rx(1, 1) = -1;
			Rx(2, 0) = 1;
		}
		if(inverse)
		{
			ret = Rx * oth;
		}
		else
		{
			ret = Rx.transpose() * oth;
		}
		
		/*
		double r = std::sqrt(vec[2]*vec[2] + vec[1]*vec[1]);
		vect_type ret;
		if(r < 1e-11)
		{
			if(vec[0] > 0){
				ret[0] = oth[0];
				ret[1] = oth[1];
				ret[2] = oth[2];
			}
			else{
				ret[0] = -oth[0];
				ret[1] = -oth[1];
				ret[2] = oth[2];
			}
		}
		else{
			double alpha = std::atan2(vec[2], vec[1]);
		
			double c = cos(alpha), s = sin(alpha);
			
			if(inverse) {
				ret[0] = vec[0] * oth[0] -        r*oth[1];
				ret[1] = vec[1] * oth[0] + vec[0]*c*oth[1] - s*oth[2];
				ret[2] = vec[2] * oth[0] + vec[0]*s*oth[1] + c*oth[2]; 
			} else {
				ret[0] = vec[0] * oth[0] + vec[1]*oth[1] + vec[2]*oth[2];
				ret[1] =     -r * oth[0] + (    c*oth[1] +      s*oth[2]) * vec[0];
				ret[2] =                      - s*oth[1] +      c*oth[2]; 
			}
		}
		*/
		return ret;
	}	
};


}  // namespace MTK


#endif /*S2_H_*/
