#ifndef USE_IKFOM_H
#define USE_IKFOM_H

#include <esekfom/esekfom.hpp>

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

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

namespace IKFoM {
    Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in);
    Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in);
    Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in);
    void h_share_model(state_ikfom &, esekfom::dyn_share_datastruct<double> &);
}

MTK::get_cov<process_noise_ikfom>::type process_noise_cov();
vect3 SO3ToEuler(const SO3 &orient);

#endif