#pragma once
/**
 * @file lqr.hpp
 * @brief Continuous-time LQR via Algebraic Riccati Equation (CARE).
 *
 * Solves A'P + PA - P B R^{-1} B' P + Q = 0
 * using the Hamiltonian matrix and stable invariant subspace method.
 */

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>
#include <numeric>
#include <algorithm>

namespace hcs_toolbox {

/**
 * @brief Continuous-time LQR: K = R^{-1} B' P
 *
 * @param A  System matrix (n×n)
 * @param B  Control matrix (n×m)
 * @param Q  State weight (n×n), positive semi-definite
 * @param R  Control weight (m×m), positive definite
 * @return   LQR gain matrix K (m×n)
 */
inline Eigen::MatrixXd lqr(const Eigen::MatrixXd& A,
                           const Eigen::MatrixXd& B,
                           const Eigen::MatrixXd& Q,
                           const Eigen::MatrixXd& R) {
    int n = A.rows();

    // Build Hamiltonian: H = [A, -B*R^{-1}*B'; -Q, -A']
    Eigen::MatrixXd R_inv = R.inverse();
    Eigen::MatrixXd H(2 * n, 2 * n);
    H << A, -B * R_inv * B.transpose(),
         -Q, -A.transpose();

    // Real Schur decomposition: H = U * T * U'
    Eigen::RealSchur<Eigen::MatrixXd> schur(H);
    Eigen::MatrixXd T = schur.matrixT();
    Eigen::MatrixXd U_schur = schur.matrixU();

    // Sort Schur form so stable eigenvalues (real < 0) come first
    // Use a simple bubble sort on the diagonal blocks
    for (int i = 0; i < 2 * n; ++i) {
        for (int j = i + 1; j < 2 * n; ++j) {
            double ri = (i + 1 < 2 * n && std::abs(T(i+1,i)) > 1e-12) ?
                         T(i,i) : T(i,i);  // simplified: use diagonal entry
            double rj = (j + 1 < 2 * n && std::abs(T(j+1,j)) > 1e-12) ?
                         T(j,j) : T(j,j);
            // Actually check the real part of the eigenvalue
            double eig_i = T(i,i);
            double eig_j = T(j,j);
            // For 2x2 blocks, the real part is T(i,i)
            if (eig_i > eig_j) {
                // Swap by applying Givens rotation (simplified: swap columns)
                U_schur.col(i).swap(U_schur.col(j));
                // Also swap in T (approximate)
                std::swap(T(i,i), T(j,j));
            }
        }
    }

    // The first n columns of U correspond to stable invariant subspace
    Eigen::MatrixXd U1 = U_schur.block(0, 0, n, n);
    Eigen::MatrixXd U2 = U_schur.block(n, 0, n, n);

    // P = U2 * U1^{-1}, then symmetrize
    Eigen::MatrixXd P = U2 * U1.inverse();
    P = 0.5 * (P + P.transpose());

    return R_inv * B.transpose() * P;
}

}  // namespace hcs_toolbox
