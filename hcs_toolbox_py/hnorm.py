"""
Canonical homogeneous norm computation using bisection method.

Original MATLAB: hnorm.m
"""

import numpy as np
from scipy.linalg import expm


def hnorm(x, Gd, P, tol=1e-6, Nmax=20):
    """
    Compute the canonical d-homogeneous norm nx induced by the weighted
    Euclidean norm sqrt(x'*P*x).

    nx > 0 such that x' * d'(-log(nx)) * P * d(-log(nx)) * x = 1  (if x is non-zero)

    Parameters
    ----------
    x : (p,) or (p,1) ndarray
        Vector whose homogeneous norm is to be computed.
    Gd : (p, p) ndarray
        Anti-Hurwitz matrix — generator of the dilation d(s) = expm(s*Gd).
    P : (p, p) ndarray
        Positive definite matrix such that P*Gd + Gd'*P > 0.
    tol : float, optional
        Computational tolerance (default 1e-6).
    Nmax : int, optional
        Maximum number of bisection iterations (default 20).

    Returns
    -------
    q : float
        The homogeneous norm nx.  Returns 0 if x is (near) zero.
    """
    x = np.asarray(x).flatten()

    nrm = np.linalg.norm(x)
    if nrm <= tol:
        return 0.0

    # --- Find lower bound a:  y = d(-a)*x has y'*P*y < 1 ---
    # For large-norm x: val > 1 at a=-1 → loop doesn't execute
    # For small-norm x: val < 1 at a=-1 → make a more negative (more dilation)
    a = -1.0

    def safe_eval(s):
        """Compute y = d(-s)*x and y'Py safely."""
        try:
            y = expm(-Gd * s) @ x
            v = y @ P @ y
            if np.isnan(v) or np.isinf(v):
                return None, None
            return y, v
        except Exception:
            return None, None

    _, val = safe_eval(a)
    if val is None:
        return float('inf') if nrm > 1 else 0.0

    while val < 1.0 and a > -746:
        a = a * 2.0
        _, val2 = safe_eval(a)
        if val2 is None:
            a = a / 2.0
            break
        val = val2

    # After loop: y'Py > 1 (or a hit limit)
    # If still val < 1: ||x||_d is extremely small
    if val < 1.0:
        return np.exp(a)

    # val > 1 → find upper bound b with val < 1
    # b > 0: y = d(-b)*x = expm(-Gd*b)*x shrinks the vector
    _, val_lower = safe_eval(a)
    if val_lower is None:
        val_lower = 1.0

    b = 1.0
    _, val = safe_eval(b)
    if val is None:
        val = val_lower

    while val > 1.0 and b < 710:
        b = b * 2.0
        _, val2 = safe_eval(b)
        if val2 is None:
            b = b / 2.0
            break
        val = val2

    # If still val > 1: ||x||_d is extremely large
    if val > 1.0:
        return float('inf')

    # val < 1 at b, val > 1 at a → bisect
    c = (a + b) / 2.0
    for i in range(Nmax):
        _, val_c = safe_eval(c)
        if val_c is None:
            break
        Qf = val_c - 1.0
        if abs(Qf) <= tol:
            break
        if Qf > 0:
            a = c
        else:
            b = c
        c = (a + b) / 2.0

    return np.exp(c)
