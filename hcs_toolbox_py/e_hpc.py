"""
Explicit evaluation of Homogeneous Proportional Controller (HPC).

Original MATLAB: e_hpc.m

    u = K0*x + nx^(1+mu) * K * d(-ln(nx)) * x

where nx = hn_fun(x) is the homogeneous norm of x.
"""

import numpy as np
from scipy.linalg import expm


def e_hpc(x, K0, K, Gd, mu, hn_fun, alpha=None, beta=None):
    """
    Evaluate the HPC control law.

        u = K0*x + nx^(1+mu) * K * expm(-ln(nx)*Gd) * x

    Parameters
    ----------
    x : (p,) or (p,1) ndarray
        State vector.
    K0 : (m, p) ndarray
        Linear component gain matrix.
    K : (m, p) ndarray
        Nonlinear component gain matrix.
    Gd : (p, p) ndarray
        Generator of dilation d(s) = expm(s*Gd).
    mu : float
        Homogeneity degree.
    hn_fun : callable
        Function that computes the homogeneous norm nx = hn_fun(x).
    alpha : float, optional
        Lower saturation bound for nx (nx ← max(alpha, nx)).
    beta : float, optional
        Upper saturation bound for nx (nx ← min(beta, nx)).

    Returns
    -------
    u : (m,) ndarray
        Control input.
    """
    x = np.asarray(x).flatten()
    m = K.shape[0]

    # Linear component
    if np.all(K0 == 0):
        u = np.zeros(m)
    else:
        u = K0 @ x

    # Apply saturation bounds to homogeneous norm
    if alpha is not None and beta is not None:
        hn = max(alpha, min(beta, hn_fun(x)))
    elif alpha is not None:
        hn = max(alpha, hn_fun(x))
    else:
        hn = hn_fun(x)

    # Nonlinear component (with numerical tolerance)
    if hn > 1e-16:
        # d(-ln(nx)) * x = expm(-ln(nx) * Gd) * x
        u = u + hn ** (1.0 + mu) * K @ (expm(-np.log(hn) * Gd) @ x)

    return u
