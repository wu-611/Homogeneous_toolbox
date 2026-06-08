"""
Upgrade Linear Observer (LO) to Homogeneous Observer (HO).

Original MATLAB: lo2ho.m

The upgrade uses duality: lo2ho(A,C,L) ≡ lpc2hpc(A',C',L') transposed.

    Linear:  dz/dt = A*z + f + L*(C*z - y)
    HO:      dz/dt = A*z + f + (L0 + |Cz-y|^(nu-1) * d(log|Cz-y|) * (L-L0)) * (Cz-y)
"""

import numpy as np
from .lpc2hpc import lpc2hpc


def lo2ho(A, C, L):
    """
    Upgrade a linear Luenberger observer to a Homogeneous Observer.

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    C : (k, n) ndarray
        Output matrix (full row rank).
    L : (n, k) ndarray
        Luenberger observer gain matrix. A + L*C must be Hurwitz.

    Returns
    -------
    L0 : (n, k) ndarray
        Gain of the linear component of HO.
    G0 : (n, n) ndarray
        Matrix defining the generator of dilation: Gd = I + nu*G0.
    nu_min : float
        Minimal admissible homogeneity degree (nu_min <= nu).
    nu_max : float
        Maximal admissible homogeneity degree (nu <= nu_max).
    """
    K0, G0, P, nu_max, nu_min = lpc2hpc(A.T, C.T, L.T)
    return K0.T, -G0.T, P, -nu_min, -nu_max
