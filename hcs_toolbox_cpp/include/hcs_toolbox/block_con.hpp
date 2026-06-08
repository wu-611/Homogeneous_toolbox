#pragma once
/**
 * @file block_con.hpp
 * @brief Canonical block controllability form.
 *
 * T*A*inv(T) has super-diagonal blocks Ai_{i+1}, T*B = [0; B0].
 *
 * Original MATLAB: block_con.m
 */

#include "trans_con.hpp"

namespace hcs_toolbox {

/**
 * @brief Transform {A,B} to canonical block controllability form.
 *
 * @param A   System matrix (n×n)
 * @param B   Control matrix (n×m)
 * @param T   [out] Transformation matrix
 * @param nt  [out] Block sizes [n1, ..., nk]
 * @param tol Rank tolerance
 * @return    true on success
 */
inline bool block_con(const Eigen::MatrixXd& A,
                      const Eigen::MatrixXd& B,
                      Eigen::MatrixXd& T,
                      std::vector<int>& nt,
                      double tol = 1e-6) {
    // Step 1: Orthogonal staircase form
    Eigen::MatrixXd T_ort;
    if (!trans_con(A, B, T_ort, nt, tol)) {
        T.resize(0, 0);
        return false;
    }

    int k = static_cast<int>(nt.size());
    int n = A.rows();

    // Build block row start indices (0-based)
    std::vector<int> ni(k);
    ni[0] = 0;
    for (int i = 1; i < k; ++i)
        ni[i] = ni[i - 1] + nt[i - 1];

    Eigen::MatrixXd A_curr = T_ort * A * T_ort.transpose();
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(n, n);

    // Step 2: Triangular transformation
    for (int i = 0; i < k - 1; ++i) {
        int r_start = ni[i];
        int r_end   = ni[i] + nt[i];
        int c_start = ni[i] + nt[i];
        int c_end   = ni[i] + nt[i] + nt[i + 1];

        // Super-diagonal block Ai_{i+1}
        Eigen::MatrixXd temp_A = A_curr.block(r_start, c_start, nt[i], nt[i + 1]);

        // Build row block to zero out lower-left elements
        Eigen::MatrixXd inv_part = (temp_A * temp_A.transpose()).inverse() * temp_A;
        Eigen::MatrixXd left_part = -inv_part * A_curr.block(r_start, 0, nt[i], c_start);

        Eigen::MatrixXd temp_block(nt[i + 1], n);
        temp_block << left_part,
                      Eigen::MatrixXd::Identity(nt[i + 1], nt[i + 1]),
                      Eigen::MatrixXd::Zero(nt[i + 1], n - c_start - nt[i + 1]);

        Eigen::MatrixXd temp_T = Eigen::MatrixXd::Identity(n, n);
        temp_T.block(ni[i + 1], 0, nt[i + 1], n) = temp_block;

        Phi = temp_T * Phi;
        A_curr = temp_T * A_curr * temp_T.inverse();
    }

    T = Phi * T_ort;
    return true;
}

}  // namespace hcs_toolbox
