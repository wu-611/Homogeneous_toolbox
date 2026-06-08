"""
Addition in the homogeneous Euclidean space.

Original MATLAB: hadd.m
"""

import numpy as np
from .hnorm import hnorm
from .hphi import hphi, hphi_inv


def hadd(x1, x2, Gd, P):
    """
    Compute the sum of vectors x1 and x2 in the homogeneous Euclidean space.

        y = Phi^{-1}( Phi(x1) + Phi(x2) )

    Parameters
    ----------
    x1 : (p,) or (p,1) ndarray
        First vector.
    x2 : (p,) or (p,1) ndarray
        Second vector.
    Gd : (p, p) ndarray
        Anti-Hurwitz matrix — generator of dilation d(s) = expm(s*Gd).
    P : (p, p) ndarray
        Positive definite matrix such that P*Gd + Gd'*P > 0.

    Returns
    -------
    y : (p,) ndarray
        Homogeneous sum of x1 and x2.
    """
    # Build the homogeneous norm function
    hn_fun = lambda v: hnorm(v, Gd, P)
    # Build a standard Euclidean norm function for the inverse mapping
    n_fun = lambda v: np.sqrt(v @ P @ v)

    y = hphi_inv(
        hphi(x1, Gd, hn_fun) + hphi(x2, Gd, hn_fun),
        Gd,
        n_fun
    )
    return y
