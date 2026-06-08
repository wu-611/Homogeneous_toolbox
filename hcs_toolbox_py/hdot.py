"""
Scalar multiplication in the homogeneous Euclidean space.

Original MATLAB: hdot.m
"""

import numpy as np
from scipy.linalg import expm


def hdot(alpha, x, Gd):
    """
    Compute the product of a scalar alpha by a vector x in the homogeneous
    Euclidean space.

        hdot(alpha, x) = sign(alpha) * expm(ln(|alpha|) * Gd) * x

    Parameters
    ----------
    alpha : float
        Scalar multiplier.
    x : (p,) or (p,1) ndarray
        Vector to be multiplied.
    Gd : (p, p) ndarray
        Anti-Hurwitz matrix — generator of dilation d(s) = expm(s*Gd).

    Returns
    -------
    y : (p,) ndarray
        Homogeneous product alpha ⊙ x.
    """
    x = np.asarray(x).flatten()
    if alpha == 0.0:
        return np.zeros_like(x)
    y = np.sign(alpha) * expm(np.log(abs(alpha)) * Gd) @ x
    return y
