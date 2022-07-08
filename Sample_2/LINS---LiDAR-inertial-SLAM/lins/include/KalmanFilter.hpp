// This file is part of LINS.
//
// Copyright (C) 2020 Chao Qin <cscharlesqin@gmail.com>,
// Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
// The Hong Kong University of Science and Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.

#ifndef INCLUDE_KALMANFILTER_HPP_
#define INCLUDE_KALMANFILTER_HPP_

#include <math_utils.h>
#include <parameters.h>
#include <use_ikfom.hpp>

#include <iostream>
#include <map>

using namespace std;
using namespace math_utils;
using namespace parameter;

namespace filter {

// GlobalState Class contains state variables including position, velocity,
// attitude, acceleration bias, gyroscope bias, and gravity
class GlobalState {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr unsigned int DIM_OF_STATE_ = 18;
  static constexpr unsigned int DIM_OF_NOISE_ = 12;
  static constexpr unsigned int pos_ = 0;
  static constexpr unsigned int vel_ = 3;
  static constexpr unsigned int att_ = 6;
  static constexpr unsigned int acc_ = 9;
  static constexpr unsigned int gyr_ = 12;
  static constexpr unsigned int gra_ = 15;

  GlobalState() { setIdentity(); }

  GlobalState(const V3D& rn, const V3D& vn, const Q4D& qbn, const V3D& ba,
              const V3D& bw) {
    setIdentity();
    rn_ = rn;
    vn_ = vn;
    qbn_ = qbn;
    ba_ = ba;
    bw_ = bw;
  }

  ~GlobalState() {}

  void setIdentity() {
    rn_.setZero();
    vn_.setZero();
    qbn_.setIdentity();
    ba_.setZero();
    bw_.setZero();
    gn_ << 0.0, 0.0, -G0;
  }

  // boxPlus operator
  void boxPlus(const Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk,
               GlobalState& stateOut) {
    stateOut.rn_ = rn_ + xk.template segment<3>(pos_);
    stateOut.vn_ = vn_ + xk.template segment<3>(vel_);
    stateOut.ba_ = ba_ + xk.template segment<3>(acc_);
    stateOut.bw_ = bw_ + xk.template segment<3>(gyr_);
    Q4D dq = axis2Quat(xk.template segment<3>(att_));
    stateOut.qbn_ = (qbn_ * dq).normalized();

    stateOut.gn_ = gn_ + xk.template segment<3>(gra_);
  }

  // boxMinus operator
  void boxMinus(const GlobalState& stateIn,
                Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk) {
    xk.template segment<3>(pos_) = rn_ - stateIn.rn_;
    xk.template segment<3>(vel_) = vn_ - stateIn.vn_;
    xk.template segment<3>(acc_) = ba_ - stateIn.ba_;
    xk.template segment<3>(gyr_) = bw_ - stateIn.bw_;
    V3D da = Quat2axis(stateIn.qbn_.inverse() * qbn_);
    xk.template segment<3>(att_) = da;

    xk.template segment<3>(gra_) = gn_ - stateIn.gn_;
  }

  GlobalState& operator=(const GlobalState& other) {
    if (this == &other) return *this;

    this->rn_ = other.rn_;
    this->vn_ = other.vn_;
    this->qbn_ = other.qbn_;
    this->ba_ = other.ba_;
    this->bw_ = other.bw_;
    this->gn_ = other.gn_;

    return *this;
  }

  // !@State
  V3D rn_;   // position in n-frame
  V3D vn_;   // velocity in n-frame
  Q4D qbn_;  // rotation from b-frame to n-frame
  V3D ba_;   // acceleartion bias
  V3D bw_;   // gyroscope bias
  V3D gn_;   // gravity
};

class StatePredictor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StatePredictor() { reset(); }

  ~StatePredictor() {}

  bool predict(double dt, const V3D& acc, const V3D& gyr) {
    if (!isInitialized()) return false;

    // if (!flag_init_imu_) {
    //   flag_init_imu_ = true;
    //   acc_last = acc;
    //   gyr_last = gyr;
    // }

    // Average acceleration and angular rate
    input_ikfom i_in;
    i_in.acc = 0.5 * (acc + acc_last);
    i_in.gyro = 0.5 * (gyr + gyr_last);
    // global_state_back = global_state;
    // Eigen::Matrix<double, 18, 1> f_ = get_f_global(global_state, i_in);
    // global_state.oplus(f_, dt);
    
    kf.predict(dt, noise_, i_in);
    covariance_ = kf.get_P();
    time_ += dt;
    acc_last = acc;
    gyr_last = gyr;
    return true;
  }

  static void calculateRPfromIMU(const V3D& acc, double& roll, double& pitch) {
    pitch = -sign(acc.z()) * asin(acc.x() / G0);
    roll = sign(acc.z()) * asin(acc.y() / G0);
  }

  void set(const state_global& state) { global_state = state; }

  void update() {
    state_ikfom update_state = kf.get_x();
    global_state.pos = global_state.pos + global_state.rot.toRotationMatrix() * update_state.pos;
    global_state.vel = global_state.rot.toRotationMatrix() * update_state.vel;
    global_state.ba = update_state.ba;
    global_state.bg = update_state.bg;
    global_state.grav = global_state.rot.toRotationMatrix() * update_state.grav;
    global_state.rot = SO3_type(global_state.rot.toRotationMatrix() * update_state.rot.toRotationMatrix());
  }

  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, const V3D& acc, const V3D& gyr,
                      double roll = 0.0, double pitch = 0.0, double yaw = 0.0) {
    state_ikfom cur_state = kf.get_x();
    cur_state.pos = rn;
    cur_state.vel = vn;
    cur_state.rot = rpy2Quat(V3D(roll, pitch, yaw)); // SO3_type(); //
    cur_state.ba = ba;
    cur_state.bg = bw;
    cur_state.grav = S2_type(0, 0, -9.81);
    kf.change_x(cur_state);
    
    time_ = time;
    acc_last = acc;
    gyr_last = gyr;
    // flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance();
  }

  void initializeCovariance(int type = 0) {
    double covX = pow(INIT_POS_STD(0), 2);
    double covY = pow(INIT_POS_STD(1), 2);
    double covZ = pow(INIT_POS_STD(2), 2);
    double covVx = pow(INIT_VEL_STD(0), 2);
    double covVy = pow(INIT_VEL_STD(1), 2);
    double covVz = pow(INIT_VEL_STD(2), 2);
    double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
    double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
    double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);

    V3D covPos = INIT_POS_STD.array().square();
    V3D covVel = INIT_VEL_STD.array().square();
    V3D covAcc = INIT_ACC_STD.array().square();
    V3D covGyr = INIT_GYR_STD.array().square();

    double peba = pow(ACC_N * ug, 2);
    double pebg = pow(GYR_N * dph, 2);
    double pweba = pow(ACC_W * ugpsHz, 2);
    double pwebg = pow(GYR_W * dpsh, 2);
    Eigen::Matrix<double, 2, 1> gra_cov(0.0001, 0.0001); //, 0.01);

    if (type == 0) {
      // Initialize using offline parameters
      covariance_.setZero();
      covariance_.block<3, 3>(0, 0) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(3, 3) =
          covVel.asDiagonal();  // vel
      covariance_.block<3, 3>(6, 6) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(9, 9) =
          covAcc.asDiagonal();  // ba
      covariance_.block<3, 3>(12, 12) =
          covGyr.asDiagonal();  // bg
      covariance_.block<2, 2>(15, 15) =
          gra_cov.asDiagonal();  // gravity
    } else if (type == 1) {
      // Inheritage previous covariance
      covariance_ = kf.get_P();
      M3D vel_cov =
          covariance_.block<3, 3>(3, 3);
      M3D acc_cov =
          covariance_.block<3, 3>(9, 9);
      M3D gyr_cov =
          covariance_.block<3, 3>(12, 12);
      Eigen::Matrix<double, 2, 2> gra_cov =
          covariance_.block<2, 2>(15, 15);

      covariance_.setZero();
      covariance_.block<3, 3>(0, 0) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(3, 3) =
          vel_cov;  // vel
      covariance_.block<3, 3>(6, 6) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(9, 9) = acc_cov;
      covariance_.block<3, 3>(12, 12) = gyr_cov;
      covariance_.block<2, 2>(15, 15) = gra_cov;
    }
    kf.change_P(covariance_);

    noise_.setZero();
    noise_.block<3, 3>(0, 0) = V3D(peba, peba, peba).asDiagonal();
    noise_.block<3, 3>(3, 3) = V3D(pebg, pebg, pebg).asDiagonal();
    noise_.block<3, 3>(6, 6) = V3D(pweba, pweba, pweba).asDiagonal();
    noise_.block<3, 3>(9, 9) = V3D(pwebg, pwebg, pwebg).asDiagonal();
  }

  void reset(int type = 0) {
    if (type == 0) {
      state_ikfom reset_state;
      state_ikfom cur_state = kf.get_x();
      reset_state.ba = cur_state.ba;
      reset_state.bg = cur_state.bg;
      reset_state.vel = cur_state.rot.toRotationMatrix().transpose() * cur_state.vel;
      reset_state.grav = cur_state.rot.toRotationMatrix().transpose() * cur_state.grav;
      initializeCovariance();
      kf.reset(reset_state, covariance_);
    } else if (type == 1) {
      V3D covPos = INIT_POS_STD.array().square();
      double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
      double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
      double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);
      covariance_ = kf.get_P();
      M3D vel_cov =
          covariance_.block<3, 3>(3, 3);
      M3D acc_cov =
          covariance_.block<3, 3>(9, 9);
      M3D gyr_cov =
          covariance_.block<3, 3>(12, 12);
      // Eigen::Matrix<double, 2, 2> gra_cov =
          // covariance_.block<2, 2>(15, 15);
      state_ikfom cur_state = kf.get_x();
      covariance_.setZero();
      covariance_.block<3, 3>(0, 0) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(3, 3) =
          cur_state.rot.toRotationMatrix().transpose() * vel_cov * cur_state.rot.toRotationMatrix();  // vel
      covariance_.block<3, 3>(6, 6) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(9, 9) = acc_cov;
      covariance_.block<3, 3>(12, 12) = gyr_cov;
      // Eigen::Matrix<double, 3, 2> Bx;
		  // cur_state.grav.S2_Bx(Bx);
      // covariance_.block<2, 2>(15, 15) =
              // Bx.transpose() * cur_state.rot.toRotationMatrix().transpose() * Bx * gra_cov * Bx.transpose() * cur_state.rot.toRotationMatrix() * Bx;

      state_ikfom reset_state;
      reset_state.ba = cur_state.ba;
      reset_state.bg = cur_state.bg;
      reset_state.vel = cur_state.rot.toRotationMatrix().transpose() * cur_state.vel;
      reset_state.grav = // global_state.rot.toRotationMatrix().transpose() * global_state.grav; // 
      cur_state.rot.toRotationMatrix().transpose() * cur_state.grav; // 
      // Eigen::Matrix<double, 3, 2> Bx_aft;
		  // reset_state.grav.S2_Bx(Bx_aft);
      // covariance_.block<2, 2>(15, 15) =
              // Bx_aft.transpose() * cur_state.rot.toRotationMatrix().transpose() * Bx * gra_cov * Bx.transpose() * cur_state.rot.toRotationMatrix() * Bx_aft;
      covariance_(15, 15) = 0.0001;
      covariance_(16, 16) = 0.0001;
      kf.reset(reset_state, covariance_);
      // initializeCovariance(1);
    }
  }

  void reset(V3D vn, V3D ba, V3D bw) {
    state_ikfom reset_state;
    reset_state.vel = vn;
    reset_state.ba = ba;
    reset_state.bg = bw;
    initializeCovariance();
    std::cout << "would not be here" << std::endl;
    kf.reset(reset_state, covariance_);
  }

  inline bool isInitialized() { return flag_init_state_; }

  // GlobalState state_;
  state_global global_state;
  // state_global global_state_back;
  esekfom::esekf<state_ikfom, 12, input_ikfom> kf;

  double time_;
  Eigen::Matrix<double, 18, 18>
      F_;
  Eigen::Matrix<double, 18, 18>
      jacobian_;
  Eigen::Matrix<double, 17, 17> covariance_;
  Eigen::Matrix<double, 12, 12>
      noise_;

  V3D acc_last;  // last acceleration measurement
  V3D gyr_last;  // last gyroscope measurement

  bool flag_init_state_;
  // bool flag_init_imu_;
};

};  // namespace filter

#endif  // INCLUDE_KALMANFILTER_HPP_
