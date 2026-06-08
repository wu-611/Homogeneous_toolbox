"""
Explicit evaluation of Fixed-time Homogeneous Proportional-Integral Controller (FHPIC).

Original MATLAB: e_fhpic.m

    FHPIC switches homogeneity degree based on x'*P*x, like FHPC.
"""

import numpy as np
from .hnorm import hnorm
from .e_hpic import e_hpic


def e_fhpic(x, K0, K, Ki, G0, mu1, mu2, P, alpha=None, beta=None):
    """
    Evaluate the FHPIC control law components (fixed-time HPIC).

    Parameters
    ----------
    x : (p,) or (p,1) ndarray
        State vector.
    K0 : (m, p) ndarray
        Linear component gain matrix.
    K : (m, p) ndarray
        Proportional nonlinear gain matrix.
    Ki : (m, p) ndarray
        Integral gain matrix.
    G0 : (p, p) ndarray
        Matrix defining generators: Gd1 = I + mu1*G0, Gd2 = I + mu2*G0.
    mu1 : float
        Local homogeneity degree (close to origin).
    mu2 : float
        Local homogeneity degree (close to infinity).
    P : (p, p) ndarray
        Positive definite matrix.
    alpha : float, optional
        Lower saturation bound for nx.
    beta : float, optional
        Upper saturation bound for nx.

    Returns
    -------
    uh : (m,) ndarray
        Homogeneous proportional component.
    ui : (m,) ndarray
        Integral component (integrant dv/dt).
    """
    x = np.asarray(x).flatten()

    # Select dilation based on weighted norm
    if x @ P @ x <= 1.0:
        Gd = np.eye(K.shape[1]) + mu1 * G0
        mu = mu1
    else:
        Gd = np.eye(K.shape[1]) + mu2 * G0
        mu = mu2

    # Build the homogeneous norm function for this Gd
    hn_fun = lambda v: hnorm(v, Gd, P)

    return e_hpic(x, K0, K, Ki, Gd, mu, hn_fun, alpha, beta)
