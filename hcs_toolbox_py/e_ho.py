"""
Explicit Euler discretization of Homogeneous Observer (HO).

Original MATLAB: e_ho.m

    dz/dt = A*z + f + (L0 + |Cz-y|^(nu-1) * d(log|Cz-y|) * L) * (Cz - y)
"""

import numpy as np
from scipy.linalg import expm


def e_ho(h, z, y, A, C, f, L0, L, Gd, nu, alpha=1e-6, beta=np.inf):
    """
    One step of explicit Euler discretization of the Homogeneous Observer.

    Parameters
    ----------
    h : float
        Sampling period.
    z : (n,) ndarray
        Current state estimate.
    y : (k,) ndarray
        Measured output.
    A : (n, n) ndarray
        System matrix.
    C : (k, n) ndarray
        Output matrix.
    f : (n,) ndarray
        Known exogenous input signal (e.g., B*u).
    L0 : (n, k) ndarray
        Homogenization gain (linear component).
    L : (n, k) ndarray
        Nonlinear observer gain (L = L_lin - L0).
    Gd : (n, n) ndarray
        Generator of dilation d(s) = expm(s*Gd).
    nu : float
        Homogeneity degree.
    alpha : float, optional
        Lower saturation bound for |Cz - y| (default 1e-6).
    beta : float, optional
        Upper saturation bound for |Cz - y| (default inf).

    Returns
    -------
    znew : (n,) ndarray
        Updated state estimate.
    """
    z = np.asarray(z).flatten()
    y = np.asarray(y).flatten()
    f = np.asarray(f).flatten()

    nCe = np.linalg.norm(C @ z - y)
    nCe = min(nCe, beta)
    nCe = max(nCe, alpha)
    nCe = max(nCe, 1e-20)

    n = A.shape[0]
    I_n = np.eye(n)

    S = L0 + expm(np.log(nCe) * (nu * I_n + Gd - I_n)) @ L
    znew = (I_n + h * (A + S @ C)) @ z + h * f - h * S @ y
    return znew
