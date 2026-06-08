#pragma once
/**
 * @file e_fhpc.hpp
 * @brief Fixed-time HPC — switches homogeneity degree at nx=1.
 *
 * Original MATLAB: e_fhpc.m
 */

#include "hnorm.hpp"
#include "e_hpc.hpp"

namespace hcs_toolbox {

inline Eigen::VectorXd e_fhpc(const Eigen::VectorXd& x,
                              const Eigen::MatrixXd& K0,
                              const Eigen::MatrixXd& K,
                              const Eigen::MatrixXd& G0,
                              double mu1,
                              double mu2,
                              const Eigen::MatrixXd& P,
                              double alpha = -1.0,
                              double beta = -1.0) {
    double mu;
    Eigen::MatrixXd Gd(K0.cols(), K0.cols());
    if ((x.transpose() * P * x).value() <= 1.0) {
        Gd = Eigen::MatrixXd::Identity(K0.cols(), K0.cols()) + mu1 * G0;
        mu = mu1;
    } else {
        Gd = Eigen::MatrixXd::Identity(K0.cols(), K0.cols()) + mu2 * G0;
        mu = mu2;
    }
    auto hn_fun = [&](const Eigen::VectorXd& v) { return hnorm(v, Gd, P); };
    return e_hpc(x, K0, K, Gd, mu, hn_fun, alpha, beta);
}

}  // namespace hcs_toolbox
