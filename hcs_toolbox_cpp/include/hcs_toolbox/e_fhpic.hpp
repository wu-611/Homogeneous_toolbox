#pragma once
/**
 * @file e_fhpic.hpp
 * @brief Fixed-time HPIC — switches homogeneity degree at nx=1.
 *
 * Original MATLAB: e_fhpic.m
 */

#include "hnorm.hpp"
#include "e_hpic.hpp"

namespace hcs_toolbox {

inline void e_fhpic(const Eigen::VectorXd& x,
                    const Eigen::MatrixXd& K0,
                    const Eigen::MatrixXd& K,
                    const Eigen::MatrixXd& Ki,
                    const Eigen::MatrixXd& G0,
                    double mu1,
                    double mu2,
                    const Eigen::MatrixXd& P,
                    Eigen::VectorXd& uh,
                    Eigen::VectorXd& ui,
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
    e_hpic(x, K0, K, Ki, Gd, mu, hn_fun, uh, ui, alpha, beta);
}

}  // namespace hcs_toolbox
