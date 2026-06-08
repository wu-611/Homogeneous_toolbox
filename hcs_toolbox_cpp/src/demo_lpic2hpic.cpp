/**
 * @file demo_lpic2hpic.cpp
 * @brief HPIC upgrade demo — rotary inverted pendulum with integral action.
 *
 * Reproduces MATLAB demo_lpic2hpic.m.
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
    // 1. Rotary Inverted Pendulum Model
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
    A(2, 2) -= km * km / Rm * B(2);
    A(3, 2) -= km * km / Rm * B(3);
    B *= km / Rm;

    // Linear feedback gains
    Eigen::RowVector4d Klin;
    Klin << 2.0, -35.0, 1.5, -3.0;
    Eigen::RowVector4d Klin_int;
    Klin_int << 0.5, -26.66, 1.26, -2.73;

    // ========================================================================
    // 2. HPIC Design
    // ========================================================================
    std::cout << "===========================================\n";
    std::cout << "demo_lpic2hpic — HPIC Upgrade (C++ version)\n";
    std::cout << "===========================================\n";

    auto params = hcs_toolbox::lpic2hpic(A, B, Klin, Klin_int);
    std::cout << "mu_min = " << params.mu_min << ", mu_max = " << params.mu_max << "\n";
    std::cout << "Ki_new = " << params.Ki_new << "\n";

    double mu = 0.16;
    Eigen::Matrix4d Gd = Eigen::Matrix4d::Identity() + mu * params.G0;
    Eigen::RowVector4d K_nl = Klin - params.K0;

    std::cout << "mu = " << mu << "\n";

    // ========================================================================
    // 3. Numerical Simulation (explicit Euler, no saturation — matches MATLAB)
    // ========================================================================
    std::cout << "Running numerical simulation...\n";

    double t = 0.0, Tmax = 4.0, h = 0.001;
    Eigen::Vector4d x(1.0, 1.0, 0.0, 0.0);
    double alpha = 0.001, beta = 1000.0;
    double v_int = 0.0, p_dist = 1.0;

    auto hn_fun = [&](const Eigen::VectorXd& v) {
        return hcs_toolbox::hnorm(v, Gd, params.P);
    };

    std::vector<double> tl, ul;
    std::vector<Eigen::Vector4d> xl;
    tl.push_back(t);
    xl.push_back(x);

    int steps = static_cast<int>(Tmax / h);
    for (int step = 0; step < steps; ++step) {
        Eigen::VectorXd uh_vec, ui_vec;
        hcs_toolbox::e_hpic(x, params.K0, K_nl, params.Ki_new, Gd, mu, hn_fun,
                            uh_vec, ui_vec, alpha, beta);

        double u = uh_vec(0) + v_int;
        v_int += h * ui_vec(0);

        // Explicit Euler + disturbance (matches MATLAB)
        x = x + h * A * x + h * B * (u + p_dist);

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

    // Output CSV
    std::ofstream csv("demo_lpic2hpic_cpp.csv");
    csv << "t,x1,x2,x3,x4,u\n";
    for (size_t i = 0; i < tl.size(); ++i) {
        csv << tl[i] << "," << xl[i](0) << "," << xl[i](1) << ","
            << xl[i](2) << "," << xl[i](3) << "," << ul[i] << "\n";
    }
    csv.close();
    std::cout << "Data saved to demo_lpic2hpic_cpp.csv\n";
    std::cout << "Plot:  python3 ../scripts/plot_demo_lpic2hpic.py\n";

    return 0;
}
