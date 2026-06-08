#pragma once
/**
 * @file hinner.hpp
 * @brief Inner product in homogeneous Euclidean space.
 *
 * <x, y>_h = Phi(x)' * P * Phi(y)
 *
 * Original MATLAB: hinner.m
 */

#include "hnorm.hpp"
#include "hphi.hpp"
#include <functional>

namespace hcs_toolbox {

inline double hinner(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& y,
                     const Eigen::MatrixXd& Gd,
                     const Eigen::MatrixXd& P) {
    if (x.norm() == 0.0 || y.norm() == 0.0)
        return 0.0;

    auto hn_fun = [&](const Eigen::VectorXd& v) { return hnorm(v, Gd, P); };
    return (hphi(x, Gd, hn_fun).transpose() * P * hphi(y, Gd, hn_fun)).value();
}

}  // namespace hcs_toolbox
