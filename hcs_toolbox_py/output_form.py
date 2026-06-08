"""
Output dynamics reduction via linear feedback.

Original MATLAB: output_form.m
"""

import numpy as np


def output_form(A, B, C):
    """
    Compute a linear feedback gain tK that reduces the dynamics of the output
    y = C*x of the linear system dx/dt = A*x + B*u to the form:

        dy/dt = As*y + Bs*v    with    u = tK*x + v

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    B : (n, m) ndarray
        Control matrix.
    C : (p, n) ndarray
        Output matrix (must have full row rank).

    Returns
    -------
    tK : (m, n) ndarray
        Feedback gain matrix.
    As : (p, p) ndarray
        Reduced system matrix for the output dynamics.
    Bs : (p, m) ndarray
        Reduced control matrix for the output dynamics.
    err : int
        Error code (0 = success, <0 = warning, >0 = error).
    """
    tol = 1e-5
    n = A.shape[1]
    p = C.shape[0]
    m = B.shape[1]

    # --- Input validation ---
    if A.shape[0] != A.shape[1]:
        print("Error: the system matrix must be square.")
        return 0, 0, 0, 2
    if B.shape[0] != A.shape[1]:
        print("Error: dimensions of system and control matrices must agree.")
        return 0, 0, 0, 3
    if C.shape[1] != A.shape[1]:
        print("Error: dimensions of system and output matrices must agree.")
        return 0, 0, 0, 4
    if C.shape[0] > A.shape[0]:
        print("Error: output matrix is too large.")
        return 0, 0, 0, 5
    if np.linalg.matrix_rank(C) < C.shape[0]:
        print("Error: output matrix must have full row rank.")
        return 0, 0, 0, 6

    # --- Compute feedback gain tK ---
    Prj = np.eye(n) - C.T @ np.linalg.inv(C @ C.T) @ C
    W = np.kron(Prj.T, C @ B)
    tmpA = -C @ A @ Prj
    zet = tmpA.flatten('F')  # Column-major (MATLAB-compatible)

    # Solve W * v_tK = zet (least squares if rectangular)
    v_tK, residuals, rank_W, s = np.linalg.lstsq(W, zet, rcond=None)

    tK = v_tK.reshape((m, n), order='F')

    # Validate the reduction
    err = 0
    if np.linalg.norm(C @ (A + B @ tK) @ Prj) > tol:
        print("Warning: the reduction seems impossible. Output model may be uncontrollable.")
        err = -1

    As = C @ (A + B @ tK) @ C.T @ np.linalg.inv(C @ C.T)
    Bs = C @ B
    return tK, As, Bs, err
