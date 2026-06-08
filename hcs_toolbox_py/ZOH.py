"""
Zero-Order Hold discretization of continuous-time linear systems.

Original MATLAB: ZOH.m
"""

import numpy as np


def ZOH(h, A, B, tol=1e-16):
    """
    Compute the Zero-Order-Hold discretization of a continuous-time linear
    system:

        dx/dt = A*x + B*u    →    x(k+1) = Ah*x(k) + Bh*u(k)

    Uses series expansion of the matrix exponential (no explicit expm call).

    Parameters
    ----------
    h : float
        Sampling period.
    A : (n, n) ndarray
        Continuous-time system matrix.
    B : (n, m) ndarray
        Continuous-time control matrix.
    tol : float, optional
        Computation precision (default 1e-16).

    Returns
    -------
    Ah : (n, n) ndarray
        Discrete-time system matrix.
    Bh : (n, m) ndarray
        Discrete-time control matrix.
    """
    n = A.shape[0]
    Ai = h * np.eye(n)
    S = np.zeros((n, n))
    i = 1

    # Series expansion: S = h*I + h^2*A/2! + h^3*A^2/3! + ...
    while np.linalg.norm(Ai, 'fro') > tol:
        S = S + Ai
        i += 1
        Ai = Ai @ A * h / i

    Bh = S @ B
    Ah = np.eye(n) + A @ S
    return Ah, Bh
