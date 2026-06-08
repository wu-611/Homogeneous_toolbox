"""
Explicit evaluation of Fixed-time Homogeneous Proportional Controller (FHPC).

Original MATLAB: e_fhpc.m

    FHPC switches between two homogeneity degrees:
        u = K0*x + nx^(1+mu1) * K * d1(-ln(nx))*x   if nx <= 1
        u = K0*x + nx^(1+mu2) * K * d2(-ln(nx))*x   if nx > 1

    where d1(s) = expm(s*Gd1), Gd1 = I + mu1*G0
          d2(s) = expm(s*Gd2), Gd2 = I + mu2*G0
"""

import numpy as np
from .hnorm import hnorm
from .e_hpc import e_hpc


def e_fhpc(x, K0, K, G0, mu1, mu2, P, alpha=None, beta=None):
    """
    Evaluate the FHPC control law (fixed-time homogeneous proportional control).

    Parameters
    ----------
    x : (p,) or (p,1) ndarray
        State vector.
    K0 : (m, p) ndarray
        Linear component gain matrix.
    K : (m, p) ndarray
        Nonlinear component gain matrix.
    G0 : (p, p) ndarray
        Matrix defining generators: Gd1 = I + mu1*G0, Gd2 = I + mu2*G0.
    mu1 : float
        Local homogeneity degree (close to origin, typically negative).
    mu2 : float
        Local homogeneity degree (close to infinity, typically positive).
    P : (p, p) ndarray
        Positive definite matrix. Gd1*P + P*Gd1' > 0 and Gd2*P + P*Gd2' > 0.
    alpha : float, optional
        Lower saturation bound for nx.
    beta : float, optional
        Upper saturation bound for nx.

    Returns
    -------
    u : (m,) ndarray
        Control input.
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

    return e_hpc(x, K0, K, Gd, mu, hn_fun, alpha, beta)
