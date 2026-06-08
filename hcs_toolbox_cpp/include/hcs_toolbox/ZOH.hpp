#pragma once
/**
 * @file ZOH.hpp
 * @brief Zero-Order Hold discretization.
 *
 * dx/dt = A*x + B*u  →  x(k+1) = Ah*x(k) + Bh*u(k)
 *
 * Original MATLAB: ZOH.m
 */

#include <Eigen/Dense>

namespace hcs_toolbox {

/**
 * @brief Zero-Order Hold discretization using series expansion.
 *
 * @param h   Sampling period
 * @param A   Continuous-time system matrix (n×n)
 * @param B   Continuous-time control matrix (n×m)
 * @param Ah  [out] Discrete-time system matrix
 * @param Bh  [out] Discrete-time control matrix
 * @param tol Computation tolerance (default 1e-16)
 */
inline void ZOH(double h,
                const Eigen::MatrixXd& A,
                const Eigen::MatrixXd& B,
                Eigen::MatrixXd& Ah,
                Eigen::MatrixXd& Bh,
                double tol = 1e-16) {
    int n = A.rows();
    Eigen::MatrixXd Ai = h * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n, n);
    int i = 1;

    while (Ai.norm() > tol) {
        S += Ai;
        ++i;
        Ai = Ai * A * h / i;
    }

    Bh = S * B;
    Ah = Eigen::MatrixXd::Identity(n, n) + A * S;
}

}  // namespace hcs_toolbox
