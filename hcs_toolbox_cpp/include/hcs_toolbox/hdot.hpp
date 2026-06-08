#pragma once
/**
 * @file hdot.hpp
 * @brief Scalar multiplication in homogeneous Euclidean space.
 *
 * hdot(alpha, x) = sign(alpha) * d(ln(|alpha|)) * x
 *
 * Original MATLAB: hdot.m
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>

namespace hcs_toolbox {

inline Eigen::VectorXd hdot(double alpha,
                            const Eigen::VectorXd& x,
                            const Eigen::MatrixXd& Gd) {
    if (alpha == 0.0)
        return Eigen::VectorXd::Zero(x.size());

    double sign = (alpha > 0.0) ? 1.0 : -1.0;
    return sign * (Gd * std::log(std::abs(alpha))).exp() * x;
}

}  // namespace hcs_toolbox
