#pragma once
/**
 * @file hproj.hpp
 * @brief Homogeneous projection onto the unit sphere.
 *
 * Finds z = d(s)*x and scalar s such that hn_fun(z) = 1.
 *
 * Original MATLAB: hproj.m
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>

namespace hcs_toolbox {

/**
 * @brief Compute homogeneous projection of x onto the unit sphere.
 *
 * @param x      Input vector (non-zero)
 * @param Gd     Generator of dilation (n×n)
 * @param hn_fun Function computing homogeneous norm: double(const Eigen::VectorXd&)
 * @param z      [out] Projected vector on unit sphere
 * @param s      [out] Dilation parameter s = -ln(nx)
 */
inline void hproj(const Eigen::VectorXd& x,
                  const Eigen::MatrixXd& Gd,
                  const std::function<double(const Eigen::VectorXd&)>& hn_fun,
                  Eigen::VectorXd& z,
                  double& s) {
    double nx = hn_fun(x);
    if (nx == 0.0) {
        throw std::runtime_error("hproj: x must be non-zero.");
    }
    s = -std::log(nx);
    z = (Gd * s).exp() * x;
}

}  // namespace hcs_toolbox
