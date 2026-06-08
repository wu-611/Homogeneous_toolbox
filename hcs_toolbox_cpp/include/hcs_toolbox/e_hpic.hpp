#pragma once
/**
 * @file e_hpic.hpp
 * @brief Explicit evaluation of Homogeneous PI Controller (HPIC).
 *
 * uh = K0*x + nx^(1+mu)*K*d(-ln(nx))*x
 * ui = nx^(1+2*mu)*Ki*d(-ln(nx))*x
 *
 * Note: MATLAB e_hpic has a bug where nargin==9 applies only alpha, not beta.
 * This implementation matches that behavior for compatibility.
 *
 * Original MATLAB: e_hpic.m
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>

namespace hcs_toolbox {

inline void e_hpic(const Eigen::VectorXd& x,
                   const Eigen::MatrixXd& K0,
                   const Eigen::MatrixXd& K,
                   const Eigen::MatrixXd& Ki,
                   const Eigen::MatrixXd& Gd,
                   double mu,
                   const std::function<double(const Eigen::VectorXd&)>& hn_fun,
                   Eigen::VectorXd& uh,
                   Eigen::VectorXd& ui,
                   double alpha = -1.0,
                   double beta = -1.0) {
    int m = K.rows();
    ui = Eigen::VectorXd::Zero(m);

    if (K0.isZero(1e-16)) {
        uh = Eigen::VectorXd::Zero(m);
    } else {
        uh = K0 * x;
    }

    // Apply saturation bounds (matches MATLAB nargin logic)
    double hn = hn_fun(x);
    // MATLAB: nargin==9 → only alpha applied (beta ignored)
    if (alpha >= 0.0)
        hn = std::max(alpha, hn);

    if (hn > 1e-20) {
        Eigen::VectorXd hpx = (Gd * (-std::log(hn))).exp() * x;
        uh += std::pow(hn, 1.0 + mu) * K * hpx;
        ui  = std::pow(hn, 1.0 + 2.0 * mu) * Ki * hpx;
    }
}

}  // namespace hcs_toolbox
