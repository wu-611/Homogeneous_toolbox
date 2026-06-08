#pragma once
/**
 * @file hnorm.hpp
 * @brief Canonical homogeneous norm using bisection method.
 *
 * Finds nx > 0 such that: x' * d'(-ln(nx)) * P * d(-ln(nx)) * x = 1
 * where d(s) = expm(s * Gd) is the dilation.
 *
 * Original MATLAB: hnorm.m
 */

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>

namespace hcs_toolbox {

/**
 * @brief Compute the canonical d-homogeneous norm of vector x.
 *
 * @param x    State vector (size n)
 * @param Gd   Generator of dilation d(s) = expm(s * Gd), anti-Hurwitz (n×n)
 * @param P    Positive definite matrix such that P*Gd + Gd'*P > 0 (n×n)
 * @param tol  Computational tolerance (default 1e-6)
 * @param Nmax Maximum bisection iterations (default 20)
 * @return     Homogeneous norm nx, or 0 if x is near zero
 */
inline double hnorm(const Eigen::VectorXd& x,
                    const Eigen::MatrixXd& Gd,
                    const Eigen::MatrixXd& P,
                    double tol = 1e-6,
                    int Nmax = 20) {
    if (x.norm() <= tol)
        return 0.0;

    double a = -1.0;
    // Search lower bound: double a negatively until y'*P*y < 1
    Eigen::VectorXd y = (Gd * (-a)).exp() * x;  // expm(Gd) since a=-1
    while ((y.transpose() * P * y).value() < 1.0 && a > -746) {
        a *= 2.0;
        y = (Gd * (-a)).exp() * x;
    }

    double c;
    double yPy = (y.transpose() * P * y).value();

    if (yPy > 1.0) {
        // Search upper bound: double b positively until y'*P*y > 1
        double b = 1.0;
        y = (Gd * (-b)).exp() * x;
        while ((y.transpose() * P * y).value() > 1.0 && b < 710) {
            b *= 2.0;
            y = (Gd * (-b)).exp() * x;
        }
        yPy = (y.transpose() * P * y).value();

        if (yPy < 1.0) {
            // Bisection
            c = (a + b) / 2.0;
            y = (Gd * (-c)).exp() * x;
            double Qf = (y.transpose() * P * y).value() - 1.0;
            int i = 0;
            while (std::abs(Qf) > tol && i < Nmax) {
                ++i;
                if (Qf > 0.0) a = c;
                else          b = c;
                c = (a + b) / 2.0;
                y = (Gd * (-c)).exp() * x;
                Qf = (y.transpose() * P * y).value() - 1.0;
            }
        } else {
            c = b;
        }
    } else {
        c = a;
    }

    return std::exp(c);
}

}  // namespace hcs_toolbox
