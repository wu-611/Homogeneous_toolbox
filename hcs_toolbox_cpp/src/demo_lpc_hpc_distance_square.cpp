/**
 * @file demo_lpc_hpc_distance_square.cpp
 * @brief HPC pursuit-evasion with formation switching (2 double-integrator agents).
 *
 * Reproduces MATLAB lpc_hpc_distance_square.m.
 */

#include "hcs_toolbox/hcs_toolbox.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

int main() {
    // ========================================================================
    // 1. System Model — Double Integrator in 2D, mass m=2
    // ========================================================================
    double m = 2.0;
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    A.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity();

    Eigen::Matrix<double, 4, 2> B_mat = Eigen::Matrix<double, 4, 2>::Zero();
    B_mat.block<2, 2>(2, 0) = Eigen::Matrix2d::Identity() / m;

    // ========================================================================
    // 2. Simulation Parameters
    // ========================================================================
    double t = 0.0, Tmax = 30.0, h = 0.01, tol_switch = 0.1;
    Eigen::Vector4d x1(1.0, 0.0, 0.0, 0.0);  // Agent 1 (target)
    Eigen::Vector4d x2(5.0, 1.0, 0.0, 0.0);  // Agent 2 (pursuer)

    // Formation points: 4 points on circle radius=1
    int m_p = 4;
    double radius = 1.0;
    std::vector<Eigen::Vector4d> dl_list(m_p);
    for (int i = 0; i < m_p; ++i) {
        double angle = 2.0 * M_PI * i / m_p;
        dl_list[i] << -radius * std::cos(angle), -radius * std::sin(angle), 0.0, 0.0;
    }

    // Find initial closest formation point
    double min_dist = std::numeric_limits<double>::max();
    int min_idx = 0;
    for (int i = 0; i < m_p; ++i) {
        double dist = (x2 - x1 - dl_list[i]).norm();
        if (dist < min_dist) { min_dist = dist; min_idx = i; }
    }
    Eigen::Vector4d d = dl_list[min_idx];

    // ========================================================================
    // 3. Initial Adaptive Linear Gain + HPC Upgrade
    // ========================================================================
    Eigen::Vector4d e = x2 - x1 - d;
    double a_coef = std::max(-m * e(2) / e(0), 1.0);
    double b_coef = std::max(-m * e(3) / e(1), 1.0);
    Eigen::Matrix2d Lambda;
    Lambda << a_coef, 0.0, 0.0, b_coef;
    Eigen::Matrix2d k2 = -2.0 * Lambda;
    Eigen::Matrix2d k1 = Lambda * (k2 + Lambda) / m;

    Eigen::Matrix<double, 2, 4> k_lin;
    k_lin << k1, k2;

    auto hpc_p = hcs_toolbox::lpc2hpc(A, B_mat, k_lin);
    double nu = hpc_p.mu_min;
    Eigen::Matrix4d Gd = Eigen::Matrix4d::Identity() + nu * hpc_p.G0;
    // Note: K0 is computed but NOT used in this specialized control law

    std::cout << "===========================================\n";
    std::cout << "demo_lpc_hpc_distance_square (C++ version)\n";
    std::cout << "===========================================\n";
    std::cout << "Initial nu = " << nu << "\n";
    std::cout << "Initial Lambda = diag(" << a_coef << ", " << b_coef << ")\n";

    // ========================================================================
    // 4. Simulation Loop
    // ========================================================================
    std::cout << "Running numerical simulation...\n";

    std::vector<double> tl;
    std::vector<Eigen::Vector4d> xl1, xl2, el;
    std::vector<Eigen::Vector2d> ul2;

    int steps = static_cast<int>(Tmax / h);
    for (int step = 0; step < steps; ++step) {
        // Agent 1: linear PD + sinusoidal reference
        Eigen::Vector2d u1 = -(Eigen::Matrix<double, 2, 4>() << Eigen::Matrix2d::Identity(),
                                Eigen::Matrix2d::Identity()).finished() * x1
                             + Eigen::Vector2d(std::sin(t), std::cos(t));
        x1 = x1 + h * (A * x1 + B_mat * u1);

        // Agent 2: HPC tracking
        e = x2 - x1 - d;
        double nx = hcs_toolbox::hnorm(e, Gd, hpc_p.P);
        double nx_sat = std::max(std::min(1.0, nx), 0.1);

        Eigen::Vector2d u2 = std::pow(nx_sat, 1.0 + nu) * k_lin
                             * ((Gd * (1.0 - std::log(nx_sat))).exp() * e);

        x2 = x2 + h * (A * x2 + B_mat * u2);

        // Formation switching check
        double cur_dist = (x2 - x1 - d).norm();
        min_dist = std::numeric_limits<double>::max();
        for (int i = 0; i < m_p; ++i) {
            double dist = (x2 - x1 - dl_list[i]).norm();
            if (dist < min_dist) { min_dist = dist; min_idx = i; }
        }

        if (min_dist + tol_switch < cur_dist) {
            d = dl_list[min_idx];
            e = x2 - x1 - d;
            a_coef = std::max(-m * e(2) / e(0), 4.0);  // min=4 after switching
            b_coef = std::max(-m * e(3) / e(1), 4.0);
            Lambda << a_coef, 0.0, 0.0, b_coef;
            k2 = -2.0 * Lambda;
            k1 = Lambda * (k2 + Lambda) / m;
            k_lin << k1, k2;
            hpc_p = hcs_toolbox::lpc2hpc(A, B_mat, k_lin);
            nu = hpc_p.mu_min;
            Gd = Eigen::Matrix4d::Identity() + nu * hpc_p.G0;
        }

        t += h;
        if (step % 5 == 0) {
            tl.push_back(t);
            xl1.push_back(x1);
            xl2.push_back(x2);
            el.push_back(e);
            ul2.push_back(u2);
        }
    }

    std::cout << "Done! Final t = " << t << "s\n";
    std::cout << "Final ||e|| = " << (x2 - x1 - d).norm() << "\n";

    // ========================================================================
    // 5. Output CSV
    // ========================================================================
    std::ofstream csv("demo_lpc_hpc_distance_square_cpp.csv");
    csv << "t,x1,y1,vx1,vy1,x2,y2,vx2,vy2,ex,ey,u2x,u2y\n";
    for (size_t i = 0; i < tl.size(); ++i) {
        csv << tl[i] << ","
            << xl1[i](0) << "," << xl1[i](1) << "," << xl1[i](2) << "," << xl1[i](3) << ","
            << xl2[i](0) << "," << xl2[i](1) << "," << xl2[i](2) << "," << xl2[i](3) << ","
            << el[i](0) << "," << el[i](1) << ","
            << ul2[i](0) << "," << ul2[i](1) << "\n";
    }
    csv.close();
    std::cout << "Data saved to demo_lpc_hpc_distance_square_cpp.csv\n";
    std::cout << "Plot:  python3 ../scripts/plot_demo_distance_square.py\n";

    return 0;
}
