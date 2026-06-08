/**
 * @file demo_lpc2hpc.cpp
 * @brief HPC upgrade demo — rotary inverted pendulum (QUBE-Servo 2).
 *
 * Reproduces MATLAB demo_lpc2hpc.m.
 *
 * Build: g++ -std=c++17 -O2 -I../include demo_lpc2hpc.cpp -o demo_lpc2hpc
 *        (requires Eigen3 with unsupported/MatrixFunctions)
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
    // 1. Rotary Inverted Pendulum Model (QUBE-Servo 2)
    // ========================================================================
    double Rm = 8.4, kt = 0.042, km = 0.042;
    double mr = 0.095, r_arm = 0.085, Jr = mr * r_arm * r_arm / 3.0, br = 1e-3;
    double mp = 0.024, Lp = 0.129, l_len = Lp / 2.0;
    double Jp = mp * Lp * Lp / 3.0, bp = 5e-5, g = 9.81;
    double Jt = Jr * Jp - mp * mp * r_arm * r_arm * l_len * l_len;

    Eigen::Matrix4d A;
    A << 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
         0.0, mp * mp * l_len * l_len * r_arm * g / Jt,
            -br * Jp / Jt, -mp * l_len * r_arm * bp / Jt,
         0.0, mp * g * l_len * Jr / Jt,
            -mp * l_len * r_arm * br / Jt, -Jr * bp / Jt;

    Eigen::Vector4d B;
    B << 0.0, 0.0, Jp / Jt, mp * l_len * r_arm / Jt;

    // Actuator dynamics
    A(2, 2) -= km * km / Rm * B(2);
    A(3, 2) -= km * km / Rm * B(3);
    B *= km / Rm;

    // Linear feedback gain (manufacturer)
    Eigen::RowVector4d Klin;
    Klin << 2.0, -35.0, 1.5, -3.0;

    // ========================================================================
    // 2. HPC Design
    // ========================================================================
    std::cout << "===========================================\n";
    std::cout << "demo_lpc2hpc — HPC Upgrade (C++ version)\n";
    std::cout << "===========================================\n";

    auto params = hcs_toolbox::lpc2hpc(A, B, Klin);
    std::cout << "mu_min = " << params.mu_min << ", mu_max = " << params.mu_max << "\n";
    std::cout << "K0 = " << params.K0 << "\n";

    double mu = -1.0;
    Eigen::Matrix4d Gd = Eigen::Matrix4d::Identity() + mu * params.G0;
    Eigen::RowVector4d K_nl = Klin - params.K0;

    std::cout << "mu = " << mu << "\n";

    // ========================================================================
    // 3. Numerical Simulation
    // ========================================================================
    std::cout << "Running numerical simulation...\n";

    double t = 0.0, Tmax = 3.0, h = 0.001;
    Eigen::Vector4d x(1.0, 1.0, 0.0, 0.0);
    double alpha = 0.1, beta = 1.0;
    int steps = static_cast<int>(Tmax / h);

    // ZOH discretization
    Eigen::Matrix4d Ah;
    Eigen::Matrix4d Bh_mat;  // will be 4x1
    Eigen::Vector4d Bh;
    {
        Eigen::MatrixXd Ah_tmp, Bh_tmp;
        hcs_toolbox::ZOH(h, A, B, Ah_tmp, Bh_tmp);
        Ah = Ah_tmp;
        Bh = Bh_tmp.col(0);
    }

    auto hn_fun = [&](const Eigen::VectorXd& v) {
        return hcs_toolbox::hnorm(v, Gd, params.P);
    };

    // Data logging (sample every 10 steps to save memory)
    std::vector<double> tl, ul;
    std::vector<Eigen::Vector4d> xl;
    tl.push_back(t);
    xl.push_back(x);

    for (int step = 0; step < steps; ++step) {
        Eigen::VectorXd u_vec = hcs_toolbox::e_hpc(x, params.K0, K_nl, Gd, mu, hn_fun, alpha, beta);
        double u = u_vec(0);

        // Control saturation (QUBE-Servo 2: ±10V)
        u = std::max(-10.0, std::min(10.0, u));

        // Plant simulation with ZOH
        x = Ah * x + Bh * u;

        t += h;
        if (step % 10 == 0) {
            tl.push_back(t);
            xl.push_back(x);
            ul.push_back(u);
        }
    }
    ul.push_back(ul.back());

    std::cout << "Done!\n";
    std::cout << "||x(Tmax)|| = " << x.norm() << "\n";

    // ========================================================================
    // 4. Output CSV for plotting
    // ========================================================================
    std::ofstream csv("demo_lpc2hpc_cpp.csv");
    csv << "t,x1,x2,x3,x4,u\n";
    for (size_t i = 0; i < tl.size(); ++i) {
        csv << tl[i] << "," << xl[i](0) << "," << xl[i](1) << ","
            << xl[i](2) << "," << xl[i](3) << "," << ul[i] << "\n";
    }
    csv.close();
    std::cout << "Data saved to demo_lpc2hpc_cpp.csv\n";
    std::cout << "Plot:  python3 ../scripts/plot_demo_lpc2hpc.py\n";

    return 0;
}
