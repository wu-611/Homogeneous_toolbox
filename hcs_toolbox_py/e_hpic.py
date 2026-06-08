"""
Explicit evaluation of Homogeneous Proportional-Integral Controller (HPIC).

Original MATLAB: e_hpic.m

    u = uh + v,   dv/dt = ui

    uh = K0*x + nx^(1+mu) * K * d(-ln(nx)) * x
    ui = nx^(1+2*mu) * Ki * d(-ln(nx)) * x
"""

import numpy as np
from scipy.linalg import expm


def e_hpic(x, K0, K, Ki, Gd, mu, hn_fun, alpha=None, beta=None):
    """
    Evaluate the HPIC control law components.

        uh = K0*x + nx^(1+mu) * K * expm(-ln(nx)*Gd) * x
        ui = nx^(1+2*mu) * Ki * expm(-ln(nx)*Gd) * x

    Parameters
    ----------
    x : (p,) or (p,1) ndarray
        State vector.
    K0 : (m, p) ndarray
        Linear component gain matrix.
    K : (m, p) ndarray
        Proportional nonlinear gain matrix.
    Ki : (m, p) ndarray
        Integral gain matrix.
    Gd : (p, p) ndarray
        Generator of dilation d(s) = expm(s*Gd).
    mu : float
        Homogeneity degree.
    hn_fun : callable
        Function that computes the homogeneous norm nx = hn_fun(x).
    alpha : float, optional
        Lower saturation bound for nx.
    beta : float, optional
        Upper saturation bound for nx.

    Returns
    -------
    uh : (m,) ndarray
        Homogeneous proportional component.
    ui : (m,) ndarray
        Integral component (integrant dv/dt).
    """
    x = np.asarray(x).flatten()
    m = K.shape[0]

    # Initialize
    ui = np.zeros(m)
    if np.all(K0 == 0):
        uh = np.zeros(m)
    else:
        uh = K0 @ x

    # Apply saturation bounds (matches MATLAB nargin logic exactly)
    # MATLAB e_hpic has 9 params → nargin==9 → only alpha is applied (beta is dead code)
    # The nargin==10 check in MATLAB is unreachable (function has only 9 params)
    if alpha is not None and beta is not None:
        # MATLAB nargin==9 path: only alpha bound is applied (beta ignored)
        hn = max(alpha, hn_fun(x))
    elif alpha is not None:
        hn = max(alpha, hn_fun(x))
    else:
        hn = hn_fun(x)

    # Nonlinear components (with numerical tolerance)
    if hn > 1e-20:
        # Homogeneous projection: hpx = d(-ln(nx)) * x
        hpx = expm(-np.log(hn) * Gd) @ x
        uh = uh + hn ** (1.0 + mu) * K @ hpx
        ui = hn ** (1.0 + 2.0 * mu) * Ki @ hpx

    return uh, ui
