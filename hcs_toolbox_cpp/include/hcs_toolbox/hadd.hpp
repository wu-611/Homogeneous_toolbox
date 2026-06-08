#pragma once
/**
 * @file hadd.hpp
 * @brief Addition in homogeneous Euclidean space.
 *
 * y = Phi^{-1}( Phi(x1) + Phi(x2) )
 *
 * Original MATLAB: hadd.m
 */

#include "hnorm.hpp"
#include "hphi.hpp"
#include <functional>

namespace hcs_toolbox {

inline Eigen::VectorXd hadd(const Eigen::VectorXd& x1,
                            const Eigen::VectorXd& x2,
                            const Eigen::MatrixXd& Gd,
                            const Eigen::MatrixXd& P) {
    auto hn_fun = [&](const Eigen::VectorXd& v) { return hnorm(v, Gd, P); };
    auto n_fun  = [&](const Eigen::VectorXd& v) { return std::sqrt((v.transpose() * P * v).value()); };

    return hphi_inv(hphi(x1, Gd, hn_fun) + hphi(x2, Gd, hn_fun), Gd, n_fun);
}

}  // namespace hcs_toolbox
