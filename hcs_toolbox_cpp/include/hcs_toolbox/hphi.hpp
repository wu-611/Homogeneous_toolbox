#pragma once
/**
 * @file hphi.hpp
 * @brief Homogeneous homeomorphism (forward and inverse).
 *
 * Phi(x) = ||x||_d * d(-ln||x||_d) * x
 * Phi^{-1}(y) = d(ln||y||) * y / ||y||
 *
 * Original MATLAB: hphi.m, hphi_inv.m
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>

namespace hcs_toolbox {

/**
 * @brief Compute the homogeneous homeomorphism Phi(x).
 */
inline Eigen::VectorXd hphi(const Eigen::VectorXd& x,
                            const Eigen::MatrixXd& Gd,
                            const std::function<double(const Eigen::VectorXd&)>& hn_fun) {
    if (x.norm() == 0.0)
        return Eigen::VectorXd::Zero(x.size());

    double hn = hn_fun(x);
    return hn * (Gd * (-std::log(hn))).exp() * x;
}

/**
 * @brief Compute the inverse homogeneous homeomorphism Phi^{-1}(y).
 */
inline Eigen::VectorXd hphi_inv(const Eigen::VectorXd& y,
                                const Eigen::MatrixXd& Gd,
                                const std::function<double(const Eigen::VectorXd&)>& n_fun) {
    if (y.norm() == 0.0)
        return Eigen::VectorXd::Zero(y.size());

    double nm = n_fun(y);
    return (Gd * std::log(nm)).exp() * y / nm;
}

}  // namespace hcs_toolbox
