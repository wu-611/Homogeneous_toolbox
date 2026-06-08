"""
Homogeneous projection onto the unit sphere.

Original MATLAB: hproj.m
"""

import numpy as np
from scipy.linalg import expm


def hproj(x, Gd, hn_fun):
    """
    Compute the homogeneous projection of a non-zero vector x onto the
    homogeneous unit sphere.

    Finds z and s such that:
        z = d(s) * x   and   hn_fun(z) = 1

    Parameters
    ----------
    x : (p,) or (p,1) ndarray
        Vector to be projected (must be non-zero).
    Gd : (p, p) ndarray
        Anti-Hurwitz matrix — generator of dilation d(s) = expm(s*Gd).
    hn_fun : callable
        Function that computes the homogeneous norm of a vector, e.g.,
        lambda v: hnorm(v, Gd, P).

    Returns
    -------
    z : (p,) ndarray
        Homogeneous projection of x (on the unit sphere).
    s : float
        Dilation parameter s = -ln(nx).
    """
    x = np.asarray(x).flatten()
    nx = hn_fun(x)
    if nx == 0.0:
        raise ValueError("Homogeneous projection: x must be non-zero.")
    s = -np.log(nx)
    z = expm(s * Gd) @ x
    return z, s
