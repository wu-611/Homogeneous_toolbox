"""
Lp norm of a signal over a time interval.

Original MATLAB: lp_norm.m
"""

import numpy as np


def lp_norm(tl, xl, a, b, p):
    """
    Compute the Lp norm of a signal x(t) sampled at times tl over the
    interval [a, b].

    Parameters
    ----------
    tl : (N,) ndarray
        Time vector (must be increasing).
    xl : (M, N) ndarray
        Signal values at each time instant (each column is a sample).
    a : float
        Start of integration interval.
    b : float
        End of integration interval.
    p : float or np.inf
        Order of the norm (p >= 1 or p == np.inf).

    Returns
    -------
    y : float
        The Lp norm of the signal over [a, b].
    """
    if a > b:
        raise ValueError("lp_norm: a must be less than or equal to b.")
    if a == b:
        return 0.0

    i = 0
    j = xl.shape[1] - 1

    # Clamp to available time range
    if tl[0] > a:
        a = tl[0]
    if b > tl[-1]:
        b = tl[-1]

    # Find first index >= a
    while tl[i] < a and i < j:
        i += 1
    # Find last index <= b
    while tl[j] > b and j > 0:
        j -= 1

    if i >= j:
        return 0.0

    if p == np.inf:
        y = np.max(np.abs(xl[:, i:j+1]))
    elif p >= 1:
        s = 0.0
        for k in range(i, j):
            dt = tl[k+1] - tl[k]
            s += dt * 0.5 * np.sum(np.abs(xl[:, k])**p + np.abs(xl[:, k+1])**p)
        y = s ** (1.0 / p)
    else:
        y = 0.0

    return y
