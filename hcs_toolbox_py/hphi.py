"""
Homogeneous homeomorphism (forward and inverse).

Original MATLAB: hphi.m, hphi_inv.m
"""

import numpy as np
from scipy.linalg import expm


def hphi(x, Gd, hn_fun):
    """
    Compute the homogeneous homeomorphism Phi(x).

        Phi(x) = ||x||_d * d(-ln||x||_d) * x

    Parameters
    ----------
    x : (p,) or (p,1) ndarray
        Input vector.
    Gd : (p, p) ndarray
        Anti-Hurwitz matrix — generator of dilation d(s) = expm(s*Gd).
    hn_fun : callable
        Function that computes the homogeneous norm of a vector.

    Returns
    -------
    y : (p,) ndarray
        Phi(x).  Zero vector if x is zero.
    """
    x = np.asarray(x).flatten()
    if np.linalg.norm(x) == 0.0:
        return np.zeros_like(x)
    hn = hn_fun(x)
    y = hn * expm(-np.log(hn) * Gd) @ x
    return y


def hphi_inv(y, Gd, n_fun):
    """
    Compute the inverse homogeneous homeomorphism Phi^{-1}(y).

        Phi^{-1}(y) = d(ln||y||) * y / ||y||

    Parameters
    ----------
    y : (p,) or (p,1) ndarray
        Input vector.
    Gd : (p, p) ndarray
        Anti-Hurwitz matrix — generator of dilation d(s) = expm(s*Gd).
    n_fun : callable
        Function that computes a norm of y (e.g., np.linalg.norm).

    Returns
    -------
    x : (p,) ndarray
        Phi^{-1}(y).  Zero vector if y is zero.
    """
    y = np.asarray(y).flatten()
    if np.linalg.norm(y) == 0.0:
        return np.zeros_like(y)
    nm = n_fun(y)
    x = expm(np.log(nm) * Gd) @ y / nm
    return x
