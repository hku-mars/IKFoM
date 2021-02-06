## IKFoM 
**IKFoM** (Iterated Kalman Filters on Manifolds) is a computationally efficient and convenient toolkit for deploying iterated Kalman filters on various robotic systems, especially systems operating on high-dimension manifold. It implements a manifold-embedding Kalman filter which separates the menifold structures from system descriptions and is able to be used by only defining the system in a canonical form and calling the respective steps accordingly. The current implementation supports the full iterated Kalman filtering for systems on manifold<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathbb{R}^m\times SO(3)\times\cdots\times SO(3)\times\mathbb{S}^2\times\cdots\times\mathbb{S}^2"> and any of its sub-manifolds, and it is extendable to other types of manifold when necessary.


**Developers**

[Dongjiao He](https://github.com/Joanna-HE)

**Our related video**: https://youtu.be/sz_ZlDkl6fA

## 1. Prerequisites

### 1.1. **Eigen && Boost**
Eigen  >= 3.3.4, Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page).

Boost >= 1.65.

## 2. Usage
Clone the repository:

```
    git clone https://github.com/hku-mars/IKFoM.git
```

1. include the necessary head file:
```
#include<esekfom/esekfom.hpp>
```
2. Select and instantiate the primitive manifolds:
```
    typedef MTK::SO3<double> SO3; // scalar type of variable: double
    typedef MTK::vect<3, double> vect3; // dimension of the defined Euclidean variable: 3
    typedef MTK::S2<double, 98, 10, 1> S2; // length of the S2 variable: 98/10; choose e1 as the original point of rotation: 1
```
3. Build system state, input and measurement as compound manifolds which are composed of the primitive manifolds:
``` 
MTK_BUILD_MANIFOLD(state, // name of compound manifold: state
((vect3, pos)) // ((primitive manifold type, name of variable))
((vect3, vel))
((SO3, rot))
((vect3, bg))
((vect3, ba))
((S2, grav))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I)) 
);
```
4. Implement the vector field <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathbf{f}\left(\mathbf{x}, \mathbf{u}, \mathbf{w}\right)"> and its differentiation <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial\mathbf{f}\left(\mathbf{x}\boxplus\delta\mathbf{x}, \mathbf{u}, \mathbf{0}\right)}{\partial\delta\mathbf{x}}">,  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial\mathbf{f}\left(\mathbf{x}, \mathbf{u}, \mathbf{w}\right)}{\partial\mathbf{w}}">:
```
Eigen::Matrix<double, state_length, 1> f(state &s, input &i)}
Eigen::Matrix<double, state_length, state_dof> df_dx(state &s, input &i)} //notice S2 has length of 3 and dimension of 2
Eigen::Matrix<double, state_length, process_noise_dof> df_dw(state &s, input &i)}
```
5. Implement the output equation <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathbf{h}\left(\mathbf{x}, \mathbf{v}\right)"> and its differentiation <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial\left(\mathbf{h}\left(\mathbf{x}\boxplus\delta\mathbf{x}, \mathbf{0}\right)\boxminus\mathbf{h}\left(\mathbf{x},\mathbf{0}\right)\right)}{\partial\delta\mathbf{x}}">, <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial\left(\mathbf{h}\left(\mathbf{x}, \mathbf{v}\right)\boxminus\mathbf{h}\left(\mathbf{x},\mathbf{0}\right)\right)}{\partial\mathbf{v}}">:
```
measurement h(state &s, bool &valid)} //the iteration stops before convergence when valid is false
Eigen::Matrix<double, measurement_dof, state_dof> dh_dx(state &s, bool &valid)} 
Eigen::Matrix<double, measurement_dof, measurement_noise_dof> dh_dv(state &s, bool &valid)}
```
6. Instantiate an **esekf** object **kf** and initialize it with initial state and covariance.
```
state init_state;
esekfom::esekf<state, process_noise_dof, input, measurement, measurement_noise_dof>::cov init_P;
esekfom::esekf<state, process_noise_dof, input, measurement, measurement_noise_dof> kf(init_state,init_P);
```
7. Deliver the defined models, maximum iteration numbers **Maximum_iter**, and the std array for testing convergence **limit** into the **esekf** object:
```
kf.init(f, df_dx, df_dw, h, dh_dx, dh_dv, Maximum_iter, limit);
```
8. In the running time, once an input **in** is received with time interval **dt**, a propagation is executed:
```
kf.predict(dt, Q, input); // process noise covariance: Q
```
9. Once a measurement **z** is received, an iterated update is executed:
```
kf.update_iterated(z, R); // measurement noise covariance: R
```
*Remarks:*
- We only show the usage when the measurement is of constant dimension and type. If the measurement of your system is changing, there are iterated update functions for the case where measurement is an Eigen vector of changing dimension, and the case where measurement is a changing manifold. The usage of those two conditions would be added later, whose principles are mostly the same as the above case.
## 3. Run the sample
Clone the repository:

```
    git clone https://github.com/hku-mars/IKFoM.git
```
In the **Samples** file folder, there is the scource code that applys the **IKFoM** on the original source code from [FAST LIO](https://github.com/hku-mars/FAST_LIO). Please follow the README.md shown in that repository excepting the step **2. Build**, which is modified as:
```
cd ~/catkin_ws/src
cp -r ~/IKFoM/Samples/FAST_LIO-stable FAST_LIO-stable
cd ..
catkin_make
source devel/setup.bash
```

## 4.Acknowledgments
Thanks for C. Hertzberg,  R.  Wagner,  U.  Frese,  and  L.  Schroder.  Integratinggeneric   sensor   fusion   algorithms   with   sound   state   representationsthrough  encapsulation  of  manifolds.

