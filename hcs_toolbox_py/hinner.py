"""
Inner product in the homogeneous Euclidean space.

Original MATLAB: hinner.m
"""

import numpy as np
from .hnorm import hnorm
from .hphi import hphi


def hinner(x, y, Gd, P):
    """
    Compute the inner product of vectors x and y in the homogeneous
    Euclidean space.

        <x, y>_h = Phi(x)' * P * Phi(y)

    Parameters
    ----------
    x : (p,) or (p,1) ndarray
        First vector.
    y : (p,) or (p,1) ndarray
        Second vector.
    Gd : (p, p) ndarray
        Anti-Hurwitz matrix — generator of dilation d(s) = expm(s*Gd).
    P : (p, p) ndarray
        Positive definite matrix such that P*Gd + Gd'*P > 0.

    Returns
    -------
    q : float
        Homogeneous inner product <x, y>_h.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if np.linalg.norm(x) == 0.0 or np.linalg.norm(y) == 0.0:
        return 0.0

    hn_fun = lambda z: hnorm(z, Gd, P)
    q = hphi(x, Gd, hn_fun) @ P @ hphi(y, Gd, hn_fun)
    return q
