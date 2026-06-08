#pragma once
/**
 * @file lpic2hpic.hpp
 * @brief Upgrade Linear PI Control (LPIC) to Homogeneous PI Control (HPIC).
 *
 * Linear:     u = K*x + v,   dv/dt = Ki*x
 * HPIC:       u = uh + v,    dv/dt = ui
 *   uh = K0*x + nx^(1+mu)*(K-K0)*d(-ln(nx))*x
 *   ui = nx^(1+2*mu)*Ki_new*d(-ln(nx))*x
 *
 * Original MATLAB: lpic2hpic.m
 */

#include "block_con.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <iostream>
#include <complex>

namespace hcs_toolbox {

struct HPICParams {
    Eigen::MatrixXd K0;
    Eigen::MatrixXd G0;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Ki_new;
    double mu_min;
    double mu_max;
};

/**
 * @brief Upgrade linear PI state feedback to HPIC.
 */
inline HPICParams lpic2hpic(const Eigen::MatrixXd& A,
                            const Eigen::MatrixXd& B,
                            const Eigen::MatrixXd& K,
                            const Eigen::MatrixXd& Ki) {
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
        throw std::runtime_error("lpic2hpic: The system is not controllable.");

    // --- Special case: B is square and invertible (rank(B) == n) ---
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_B(B);
    int rank_B = (svd_B.singularValues().array() > tol).count();
    if (rank_B < m)
        throw std::runtime_error("lpic2hpic: Control matrix must have full column rank.");

    // --- Stability margin check ---
    Eigen::MatrixXd M_aug(n + m, n + m);
    M_aug << A + B * K, B,
             Ki,         Eigen::MatrixXd::Zero(m, m);
    Eigen::VectorXcd eig_aug = M_aug.eigenvalues();
    double max_real = -std::numeric_limits<double>::max();
    for (int i = 0; i < eig_aug.size(); ++i)
        max_real = std::max(max_real, eig_aug[i].real());
    double rho = -max_real;
    if (rho < tol)
        throw std::runtime_error("lpic2hpic: Insufficient stability margin.");
    rho = std::min(rho, 1.0);

    // --- Special case: B square and invertible ---
    if (rank_B == n) {
        HPICParams p;
        p.K0 = -B.transpose() * (B * B.transpose()).inverse() * A;
        p.G0 = Eigen::MatrixXd::Zero(n, n);
        p.P = Eigen::MatrixXd::Identity(n, n);
        p.Ki_new = Ki;
        p.mu_min = -0.5;
        p.mu_max = 1.0;
        return p;
    }

    // --- Block decomposition ---
    Eigen::MatrixXd T_mat;
    std::vector<int> nt;
    if (!block_con(A, B, T_mat, nt))
        throw std::runtime_error("lpic2hpic: Block decomposition failed.");

    int k = static_cast<int>(nt.size());
    std::vector<int> n_ind(k);
    n_ind[0] = 0;
    for (int i = 1; i < k; ++i)
        n_ind[i] = n_ind[i - 1] + nt[i - 1];

    Eigen::MatrixXd Anew = T_mat * A * T_mat.inverse();
    Eigen::MatrixXd Bnew = T_mat * B;

    // --- K0 ---
    Eigen::MatrixXd B0 = Bnew.block(n_ind[k - 1], 0, nt[k - 1], m);
    Eigen::MatrixXd K0_new = -B0.transpose() * (B0 * B0.transpose()).inverse()
                             * Anew.block(n_ind[k - 1], 0, nt[k - 1], n);
    K0_new = K0_new * T_mat;

    // --- G0 ---
    Eigen::VectorXd vG0(n);
    for (int i = 0; i < k; ++i)
        vG0.segment(n_ind[i], nt[i]).setConstant(k - 1 - i);
    Eigen::MatrixXd G0 = -T_mat.inverse() * vG0.asDiagonal() * T_mat;

    // --- Compute P ---
    Eigen::MatrixXd At = A + B * K;
    Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd I_m = Eigen::MatrixXd::Identity(m, m);

    Eigen::MatrixXd P12 = -0.5 * B * (B.transpose() * B).inverse();

    // Eigenvalue check (uses A, not At — matches MATLAB lpic2hpic line 123)
    Eigen::MatrixXd M_check = -P12 * Ki - Ki.transpose() * P12.transpose()
                              + A.transpose() * P12 * P12.transpose() * A;
    Eigen::VectorXcd eig_check = M_check.eigenvalues();
    double lam = 0.0;
    for (int i = 0; i < eig_check.size(); ++i)
        lam = std::max(lam, -eig_check[i].real());

    // Lyapunov equation: kron(I, At') + kron(At', I) * vec(P) = -vec(Q1)
    Eigen::MatrixXd W1(n * n, n * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i == j)
                W1.block(i * n, j * n, n, n) = At.transpose();
            else
                W1.block(i * n, j * n, n, n) = Eigen::MatrixXd::Zero(n, n);
            W1.block(i * n, j * n, n, n) += At.transpose()(i, j) * I_n;
        }
    }
    // Q1 uses A, not At — matches MATLAB lpic2hpic line 126
    Eigen::MatrixXd Q1 = (2.0 + lam) * I_n + A.transpose() * P12 * P12.transpose() * A
                         - P12 * Ki - Ki.transpose() * P12.transpose();
    Eigen::VectorXd zet1(n * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            zet1(j * n + i) = -Q1(i, j);  // column-major
    Eigen::VectorXd v_P1 = W1.colPivHouseholderQr().solve(zet1);
    Eigen::MatrixXd P(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            P(i, j) = v_P1(j * n + i);

    // Validate P
    Eigen::VectorXcd eig_P = P.eigenvalues();
    double max_eig_P = -std::numeric_limits<double>::max();
    for (int i = 0; i < eig_P.size(); ++i)
        max_eig_P = std::max(max_eig_P, eig_P[i].real());
    if (max_eig_P < tol)
        throw std::runtime_error("lpic2hpic: Upgrade impossible (P not positive definite).");

    Eigen::VectorXcd eig_PAt = (P * At + At.transpose() * P).eigenvalues();
    double max_eig_PAt = -std::numeric_limits<double>::max();
    for (int i = 0; i < eig_PAt.size(); ++i)
        max_eig_PAt = std::max(max_eig_PAt, eig_PAt[i].real());
    if (max_eig_PAt > -tol)
        throw std::runtime_error("lpic2hpic: Upgrade impossible (Lyapunov condition fails).");

    // --- Compute Ki_new via P2 ---
    // Build W3 = kron(Ki', I_m): Ki' is (n,m), I_m is (m,m), result is (n*m, m*m)
    Eigen::MatrixXd W3(n * m, m * m);
    for (int r_k = 0; r_k < n; ++r_k)       // row in Ki'
        for (int r_i = 0; r_i < m; ++r_i)   // row in I_m
            for (int c_k = 0; c_k < m; ++c_k)   // col in Ki'
                for (int c_i = 0; c_i < m; ++c_i)  // col in I_m
                    W3(r_k * m + r_i, c_k * m + c_i) = Ki(c_i, r_k) * (r_i == c_k ? 1.0 : 0.0);

    // Build W4 = kron(I_m, Ki'): I_m is (m,m), Ki' is (n,m), result is (n*m, m*m)
    Eigen::MatrixXd W4(n * m, m * m);
    for (int r_i = 0; r_i < m; ++r_i)       // row in I_m
        for (int r_k = 0; r_k < n; ++r_k)   // row in Ki'
            for (int c_i = 0; c_i < m; ++c_i)   // col in I_m
                for (int c_k = 0; c_k < m; ++c_k)  // col in Ki'
                    W4(r_i * n + r_k, c_i * m + c_k) = (r_i == c_i ? Ki(c_k, r_k) : 0.0);

    Eigen::MatrixXd W_stacked(2 * m * n, m * m);
    W_stacked << W3, W4;

    // zet3 = vec(-B'*P) in column-major order
    Eigen::MatrixXd BtP = -B.transpose() * P;  // (m, n)
    Eigen::VectorXd zet3(m * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            zet3(j * m + i) = BtP(i, j);

    // zet4 = vec(-P*B) in column-major order
    Eigen::MatrixXd PB_mat = -P * B;  // (n, m)
    Eigen::VectorXd zet4(n * m);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; ++i)
            zet4(j * n + i) = PB_mat(i, j);

    Eigen::VectorXd zet_stacked(zet3.size() + zet4.size());
    zet_stacked << zet3, zet4;

    Eigen::VectorXd v_P2 = W_stacked.colPivHouseholderQr().solve(zet_stacked);
    Eigen::MatrixXd P2(m, m);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i)
            P2(i, j) = v_P2(j * m + i);
    P2 = (P2 + P2.transpose()) / 2.0;

    Eigen::VectorXcd eig_P2 = P2.eigenvalues();
    double max_eig_P2 = -std::numeric_limits<double>::max();
    for (int i = 0; i < eig_P2.size(); ++i)
        max_eig_P2 = std::max(max_eig_P2, eig_P2[i].real());
    if (max_eig_P2 < tol)
        throw std::runtime_error("lpic2hpic: Upgrade impossible (P2 not positive definite).");

    Eigen::MatrixXd Ki_new = -P2.inverse() * B.transpose() * P;

    // --- Admissible mu range ---
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

    HPICParams params;
    params.K0 = K0_new;
    params.G0 = G0;
    params.P = P;
    params.Ki_new = Ki_new;

    if (lambda_max > tol)
        params.mu_min = std::max(-0.5, -1.0 / lambda_max + tol);
    else
        params.mu_min = -0.5;

    if (lambda_min < -tol)
        params.mu_max = std::min(1.0 / k, -1.0 / lambda_min);
    else
        params.mu_max = 1.0 / k;

    return params;
}

}  // namespace hcs_toolbox
