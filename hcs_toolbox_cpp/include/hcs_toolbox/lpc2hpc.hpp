#pragma once
/**
 * @file lpc2hpc.hpp
 * @brief Upgrade Linear Proportional Control (LPC) to Homogeneous Proportional Control (HPC).
 *
 * Linear:     u = K*x
 * HPC:        u = K0*x + nx^(1+mu)*(K-K0)*d(-ln(nx))*x
 *
 * Original MATLAB: lpc2hpc.m
 */

#include "block_con.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <iostream>
#include <complex>

namespace hcs_toolbox {

struct HPCParams {
    Eigen::MatrixXd K0;
    Eigen::MatrixXd G0;
    Eigen::MatrixXd P;
    double mu_min;
    double mu_max;
};

/**
 * @brief Upgrade linear state feedback to HPC.
 *
 * @param A   System matrix (n×n)
 * @param B   Control matrix (n×m)
 * @param K   Linear feedback gain (m×n), A+B*K must be Hurwitz
 * @return    HPC parameters {K0, G0, P, mu_min, mu_max}
 */
inline HPCParams lpc2hpc(const Eigen::MatrixXd& A,
                         const Eigen::MatrixXd& B,
                         const Eigen::MatrixXd& K) {
    double tol = 1e-5;
    int n = A.rows();
    int m = B.cols();

    // --- Controllability check ---
    Eigen::MatrixXd U_ctrb(n, n * m);
    Eigen::MatrixXd tA = Eigen::MatrixXd::Identity(n, n);
    for (int i = 0; i < n; ++i) {
        U_ctrb.middleCols(i * m, m) = tA * B;
        tA = tA * A;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_ctrb(U_ctrb);
    if ((svd_ctrb.singularValues().array() > tol).count() < n)
        throw std::runtime_error("lpc2hpc: The system is not controllable.");

    if ((U_ctrb * U_ctrb.transpose()).determinant() < tol)
        std::cerr << "Warning: Weakly controllable. Parameters may be badly tuned.\n";

    // --- Stability margin check ---
    Eigen::MatrixXd Acl = A + B * K;
    Eigen::VectorXcd eig_cl = Acl.eigenvalues();
    double max_real = -std::numeric_limits<double>::max();
    for (int i = 0; i < eig_cl.size(); ++i)
        max_real = std::max(max_real, eig_cl[i].real());
    double rho = -max_real * 0.001;
    if (rho < tol)
        throw std::runtime_error("lpc2hpc: Insufficient stability margin. Upgrade impossible.");

    // --- Special case: B is square and invertible (rank(B) == n) ---
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_B(B);
    int rank_B = (svd_B.singularValues().array() > tol).count();
    if (rank_B == n) {
        HPCParams p;
        p.K0 = -B.transpose() * (B * B.transpose()).inverse() * A;
        p.G0 = Eigen::MatrixXd::Identity(n, n);
        p.P = Eigen::MatrixXd::Identity(n, n);
        p.mu_min = -1.0;
        p.mu_max = 1.0;
        return p;
    }

    // --- Block decomposition ---
    Eigen::MatrixXd T;
    std::vector<int> nt;
    if (!block_con(A, B, T, nt))
        throw std::runtime_error("lpc2hpc: Block decomposition failed.");

    int k = static_cast<int>(nt.size());
    std::vector<int> n_ind(k);
    n_ind[0] = 0;
    for (int i = 1; i < k; ++i)
        n_ind[i] = n_ind[i - 1] + nt[i - 1];

    Eigen::MatrixXd Anew = T * A * T.inverse();
    Eigen::MatrixXd Bnew = T * B;

    // --- Compute K0 ---
    Eigen::MatrixXd B0 = Bnew.block(n_ind[k - 1], 0, nt[k - 1], m);
    Eigen::MatrixXd K0_new = -B0.transpose() * (B0 * B0.transpose()).inverse()
                             * Anew.block(n_ind[k - 1], 0, nt[k - 1], n);
    K0_new = K0_new * T;

    // --- Compute G0 ---
    Eigen::VectorXd vG0(n);
    for (int i = 0; i < k; ++i)
        vG0.segment(n_ind[i], nt[i]).setConstant(k - 1 - i);
    Eigen::MatrixXd G0 = -T.inverse() * vG0.asDiagonal() * T;

    // --- Compute P from Lyapunov equation (A+BK)'*P + P*(A+BK) = -2*I ---
    Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(n, n);
    // kron(I, Acl') + kron(Acl', I)
    Eigen::MatrixXd W0(n * n, n * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i == j)
                W0.block(i * n, j * n, n, n) = Acl.transpose();
            else
                W0.block(i * n, j * n, n, n) = Eigen::MatrixXd::Zero(n, n);
            W0.block(i * n, j * n, n, n) += Acl.transpose()(i, j) * I_n;
        }
    }
    Eigen::VectorXd zet0 = -2.0 * Eigen::VectorXd::Map(I_n.data(), n * n);
    // Reshape column-major (MATLAB compatible): I_n(:) is column-major
    // The Kronecker formulation uses column-major vec
    Eigen::VectorXd v_P = W0.colPivHouseholderQr().solve(zet0);
    Eigen::MatrixXd P = Eigen::Map<Eigen::MatrixXd>(v_P.data(), n, n).transpose();

    // --- Compute admissible mu range ---
    Eigen::MatrixXd sqrtP = P.sqrt();
    Eigen::MatrixXd inv_sqrtP = sqrtP.inverse();
    Eigen::MatrixXd M_sym = sqrtP * G0 * inv_sqrtP + inv_sqrtP * G0.transpose() * sqrtP;
    Eigen::VectorXcd lambdas = M_sym.eigenvalues();
    double lambda_min = std::numeric_limits<double>::max();
    double lambda_max = -std::numeric_limits<double>::max();
    for (int i = 0; i < lambdas.size(); ++i) {
        lambda_min = std::min(lambda_min, lambdas[i].real());
        lambda_max = std::max(lambda_max, lambdas[i].real());
    }

    HPCParams params;
    params.K0 = K0_new;
    params.G0 = G0;
    params.P = P;

    if (lambda_max > tol)
        params.mu_min = std::max(-1.0, -1.0 / lambda_max + tol);
    else
        params.mu_min = -1.0;

    if (lambda_min < -tol)
        params.mu_max = std::min(1.0 / k, -1.0 / lambda_min);
    else
        params.mu_max = 1.0 / k;

    return params;
}

}  // namespace hcs_toolbox
