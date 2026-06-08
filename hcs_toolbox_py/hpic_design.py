"""
Direct HPIC design (Homogeneous Proportional-Integral Control).

Original MATLAB: hpic_design.m

    u = uh + v,   dv/dt = ui

    uh = K0*x + nx^(1+mu) * K * d(-ln(nx)) * x
    ui = nx^(1+2*mu) * Ki * d(-ln(nx)) * x / (x' * d'(-ln(nx)) * P * Gd * d(-ln(nx)) * x)
"""

from .hpc_design import hpc_design
import numpy as np


def hpic_design(A, B, mu=-1.0, rho=1.0, gamma=0.0):
    """
    Design HPIC parameters directly.

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    B : (n, m) ndarray
        Control matrix.
    mu : float, optional
        Homogeneity degree (default -1).
    rho : float, optional
        Convergence time tuning parameter (default 1).
    gamma : float, optional
        Matched perturbation bound (default 0).

    Returns
    -------
    K0 : (m, n) ndarray
        Linear component gain.
    K : (m, n) ndarray
        Proportional nonlinear gain.
    Ki : (m, n) ndarray
        Integral gain.
    Gd : (n, n) ndarray
        Generator of dilation.
    P : (n, n) ndarray
        Positive definite matrix for homogeneous norm.
    """
    K0, K, Gd, P = hpc_design(A, B, mu, rho, gamma)
    Ki = -B.T @ P
    return K0, K, Ki, Gd, P
