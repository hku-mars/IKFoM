#ifndef USE_IKFOM_HPP_
#define USE_IKFOM_HPP_

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3_type;
typedef MTK::S2<double, 98100, 10000, 1> S2_type; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos))
((vect3, vel))
((SO3_type, rot))
((vect3, ba))
((vect3, bg))
((S2_type, grav))
);

MTK_BUILD_MANIFOLD(state_global,
((vect3, pos))
((vect3, vel))
((SO3_type, rot))
((vect3, ba))
((vect3, bg))
((S2_type, grav))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, na))
((vect3, ng))
((vect3, nba))
((vect3, nbg))
);

MTK::get_cov<process_noise_ikfom>::type process_noise_cov();

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 18, 1> get_f(state_ikfom &s, const input_ikfom &in);

// Eigen::Matrix<double, 18, 1> get_f_global(state_global &s, const input_ikfom &in);

Eigen::Matrix<double, 18, 17> df_dx(state_ikfom &s, const input_ikfom &in);

Eigen::Matrix<double, 18, 12> df_dw(state_ikfom &s, const input_ikfom &in);

vect3 SO3ToEuler(const SO3_type &orient); 

#endif  // 
