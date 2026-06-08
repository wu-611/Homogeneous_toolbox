#pragma once
/**
 * @file e_hpc.hpp
 * @brief Explicit evaluation of Homogeneous Proportional Controller (HPC).
 *
 * u = K0*x + nx^(1+mu)*K*d(-ln(nx))*x
 *
 * Original MATLAB: e_hpc.m
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>

namespace hcs_toolbox {

inline Eigen::VectorXd e_hpc(const Eigen::VectorXd& x,
                             const Eigen::MatrixXd& K0,
                             const Eigen::MatrixXd& K,
                             const Eigen::MatrixXd& Gd,
                             double mu,
                             const std::function<double(const Eigen::VectorXd&)>& hn_fun,
                             double alpha = -1.0,
                             double beta = -1.0) {
    int m = K.rows();
    Eigen::VectorXd u;
    if (K0.isZero(1e-16))
        u = Eigen::VectorXd::Zero(m);
    else
        u = K0 * x;

    double hn = hn_fun(x);
    if (alpha >= 0.0 && beta >= 0.0)
        hn = std::max(alpha, std::min(beta, hn));
    else if (alpha >= 0.0)
        hn = std::max(alpha, hn);

    if (hn > 1e-16)
        u += std::pow(hn, 1.0 + mu) * K * ((Gd * (-std::log(hn))).exp() * x);

    return u;
}

}  // namespace hcs_toolbox
