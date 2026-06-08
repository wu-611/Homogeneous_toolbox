#pragma once
/**
 * @file trans_con.hpp
 * @brief Orthogonal transformation to block controllability staircase form.
 *
 * Original MATLAB: trans_con.m
 */

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>

namespace hcs_toolbox {

/**
 * @brief Orthogonal staircase transformation for controllable pair {A, B}.
 *
 * @param A   System matrix (n×n)
 * @param B   Control matrix (n×m)
 * @param T   [out] Orthogonal transformation matrix, or empty if uncontrollable
 * @param nt  [out] Block sizes [n1, ..., nk]
 * @param tol Rank tolerance (default 1e-6)
 * @return    true on success, false if uncontrollable
 */
inline bool trans_con(const Eigen::MatrixXd& A,
                      const Eigen::MatrixXd& B,
                      Eigen::MatrixXd& T,
                      std::vector<int>& nt,
                      double tol = 1e-6) {
    int n = A.rows();

    // Controllability check
    Eigen::MatrixXd U(n, n * B.cols());
    Eigen::MatrixXd Ak = Eigen::MatrixXd::Identity(n, n);
    for (int i = 0; i < n; ++i) {
        U.middleCols(i * B.cols(), B.cols()) = Ak * B;
        Ak = Ak * A;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_ctrb(U, Eigen::ComputeThinU);
    if ((svd_ctrb.singularValues().array() > tol).count() < n) {
        T.resize(0, 0);
        nt.clear();
        return false;
    }

    // Staircase orthogonal transformation
    T = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd Ak_mat = A;
    Eigen::MatrixXd Bk = B;
    int l = 0;
    nt.clear();

    while (true) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Bk, Eigen::ComputeThinU | Eigen::ComputeThinV);
        int rk = (svd.singularValues().array() > tol).count();

        if (rk >= Ak_mat.rows()) {
            nt.insert(nt.begin(), rk);
            break;
        }
        nt.insert(nt.begin(), rk);

        int nk = Ak_mat.rows();

        // B_ort = null(Bk')' — rows orthogonal to column space of Bk
        Eigen::MatrixXd Bk_t = Bk.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_Bkt(Bk_t, Eigen::ComputeFullV);
        // null space of Bk': columns of V corresponding to zero singular values
        Eigen::MatrixXd B_ort = svd_Bkt.matrixV().rightCols(nk - rk).transpose();

        // B_p = null(B_ort)' — complement
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_Bort(B_ort, Eigen::ComputeFullV);
        Eigen::MatrixXd B_p = svd_Bort.matrixV().rightCols(rk).transpose();

        if (nk < n) {
            Eigen::MatrixXd T_new = Eigen::MatrixXd::Identity(n, n);
            // Top-left: [B_ort; B_p]
            Eigen::MatrixXd top_left(nk, nk);
            top_left << B_ort, B_p;
            T_new.topLeftCorner(nk, nk) = top_left;
            // Bottom-right already identity (handles the l offset)
            // Actually we need to handle this more carefully
            // Partition: [top_left  zeros(nk,l); zeros(l,nk) eye(l)]
            T_new.block(0, 0, nk, nk) = top_left;
            T_new.block(0, nk, nk, l).setZero();
            T_new.block(nk, 0, l, nk).setZero();
            // bottom-right eye(l) already set by identity
            T = T_new * T;
        } else {
            Eigen::MatrixXd T_new(nk, nk);
            T_new << B_ort, B_p;
            T = T_new;
        }

        l += rk;
        Bk = B_ort * Ak_mat * B_p.transpose();
        Ak_mat = B_ort * Ak_mat * B_ort.transpose();
    }

    return true;
}

}  // namespace hcs_toolbox
