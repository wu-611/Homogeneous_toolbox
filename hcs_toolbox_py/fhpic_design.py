"""
Direct FHPIC design (Fixed-time Homogeneous Proportional-Integral Control).

Original MATLAB: fhpic_design.m

    u = uh + v,   dv/dt = ui
    uh = FHPC,   ui = integrant with x'*P*x switching
"""

from .fhpc_design import fhpc_design
import numpy as np


def fhpic_design(A, B):
    """
    Design FHPIC parameters for fixed-time stabilization with integral action.

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    B : (n, m) ndarray
        Control matrix.

    Returns
    -------
    K0 : (m, n) ndarray
        Linear component gain.
    K : (m, n) ndarray
        Proportional nonlinear gain.
    Ki : (m, n) ndarray
        Integral gain.
    G0 : (n, n) ndarray
        Matrix defining generators.
    P : (n, n) ndarray
        Positive definite matrix for homogeneous norm.
    mu1 : float
        Negative homogeneity degree.
    mu2 : float
        Positive homogeneity degree.
    """
    K0, K, G0, P, mu1, mu2 = fhpc_design(A, B)[:6]
    mu1 = max(mu1, -0.5)
    Ki = -B.T @ P
    return K0, K, Ki, G0, P, mu1, mu2
