"""
Upgrade Linear Proportional-Integral Control (LPIC) to Homogeneous
Proportional-Integral Control (HPIC).

Original MATLAB: lpic2hpic.m

The upgrade:
    Linear:   u = K*x + v,   dv/dt = Ki*x
    HPIC:     u = uh + v,    dv/dt = ui

    uh = K0*x + nx^(1+mu) * (K - K0) * px
    ui = nx^(1+2*mu) * Ki_new * px / (px' * P * Gd * px)

where px = d(-ln(nx))*x is the homogeneous projection.
"""

import numpy as np
from scipy.linalg import sqrtm
from .block_con import block_con


def lpic2hpic(A, B, K, Ki):
    """
    Upgrade a linear PI controller to a Homogeneous PI Controller (HPIC).

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    B : (n, m) ndarray
        Control matrix (must have full column rank).
    K : (m, n) ndarray
        Proportional gain matrix. A + B*K must be Hurwitz.
    Ki : (m, n) ndarray
        Integral gain matrix.

    Returns
    -------
    K0 : (m, n) ndarray
        Feedback gain of the linear component of HPC.
    G0 : (n, n) ndarray
        Matrix defining the generator of dilation: Gd = I + mu*G0.
    P : (n, n) ndarray
        Positive definite matrix defining the canonical homogeneous norm.
    Ki_new : (m, n) ndarray
        Redesigned integral gain for HPIC.
    mu_min : float
        Minimal admissible homogeneity degree.
    mu_max : float
        Maximal admissible homogeneity degree.
    """
    tol = 1e-5
    n = A.shape[0]
    m = B.shape[1]

    # --- Controllability check ---
    U_ctrb = B.copy()
    tA = A.copy()
    k_ctrb = 0
    if np.linalg.matrix_rank(U_ctrb) == n:
        k_ctrb = 1
    i = 1
    while (k_ctrb == 0) and (i <= n - 1):
        i += 1
        U_ctrb = np.hstack([U_ctrb, tA @ B])
        if np.linalg.matrix_rank(U_ctrb) == n:
            k_ctrb = i
        tA = tA @ A

    if k_ctrb == 0:
        raise ValueError("lpic2hpic: The system is not controllable.")

    if np.linalg.matrix_rank(B) < m:
        raise ValueError("lpic2hpic: The control matrix must have full column rank.")

    # --- Stability margin check (augmented system) ---
    # Check eigenvalues of [A+B*K  B; Ki  0]
    M_aug = np.block([[A + B @ K, B],
                      [Ki, np.zeros((m, m))]])
    rho = -max(np.real(np.linalg.eigvals(M_aug)))
    if rho < tol:
        raise ValueError("lpic2hpic: The linear control system does not have "
                         "sufficient stability margin.")
    rho = min(rho, 1.0)

    # --- Special case: B is square and invertible ---
    if np.linalg.matrix_rank(B) == n:
        K0 = -B.T @ np.linalg.inv(B @ B.T) @ A
        G0 = np.zeros((n, n))
        P = np.eye(n)
        Ki_new = Ki.copy()
        mu_min = -0.5
        mu_max = 1.0
        return K0, G0, P, Ki_new, mu_min, mu_max

    # --- Block decomposition ---
    T, nt = block_con(A, B)
    if isinstance(T, int) and T == 0:
        raise ValueError("lpic2hpic: Block decomposition failed.")

    Anew = T @ A @ np.linalg.inv(T)
    k = len(nt)

    # Build block row start indices (0-based)
    n_ind = [0]
    s = 0
    for i in range(k - 1):
        s += nt[i]
        n_ind.append(s)

    Bnew = T @ B
    B0 = Bnew[n_ind[-1]:n, :m]
    K0 = -B0.T @ np.linalg.inv(B0 @ B0.T) @ Anew[n_ind[-1]:n, :n]
    K0 = K0 @ T

    # --- Compute G0 ---
    vG0 = []
    for i in range(k):
        vG0.extend([(k - 1 - i)] * nt[i])
    G0 = -np.linalg.inv(T) @ np.diag(vG0) @ T

    # --- Compute P via Lyapunov-like equation ---
    At = A + B @ K
    I_n = np.eye(n)
    I_m = np.eye(m)

    P12 = -0.5 * B @ np.linalg.inv(B.T @ B)

    # Check eigenvalues (uses A, NOT At = A+B*K — matches MATLAB lpic2hpic line 123)
    lambdas_check = np.real(np.linalg.eigvals(
        -P12 @ Ki - Ki.T @ P12.T + A.T @ P12 @ P12.T @ A))
    lam = max(0.0, -np.min(lambdas_check))

    # Solve Lyapunov equation for P1 (reshaped)
    W1 = np.kron(I_n, At.T) + np.kron(At.T, I_n)
    Q1 = (2.0 + lam) * I_n + A.T @ P12 @ P12.T @ A - P12 @ Ki - Ki.T @ P12.T
    zet1 = -Q1.flatten('F')
    v_P1 = np.linalg.solve(W1, zet1)
    P = v_P1.reshape((n, n), order='F')

    if max(np.real(np.linalg.eigvals(P))) < tol:
        raise ValueError("lpic2hpic: Upgrade impossible (P not positive definite).")

    if max(np.real(np.linalg.eigvals(P @ At + At.T @ P))) > -tol:
        raise ValueError("lpic2hpic: Upgrade impossible (Lyapunov condition fails).")

    # --- Compute Ki_new ---
    # Solve for P2 from coupled equations:  Ki'*P2 = -B'*P  and  P2*Ki = -P*B
    # Using least squares on the stacked system
    W3 = np.kron(Ki.T, I_m)    # kron(Ki', I_m)
    zet3 = (-B.T @ P).flatten('F')
    W4 = np.kron(I_m, Ki.T)    # kron(I_m, Ki')
    zet4 = (-P @ B).flatten('F')

    W_stacked = np.vstack([W3, W4])
    zet_stacked = np.hstack([zet3, zet4])
    v_P2, _, _, _ = np.linalg.lstsq(W_stacked, zet_stacked, rcond=None)
    P2 = v_P2.reshape((m, m), order='F')
    P2 = (P2 + P2.T) / 2.0

    if max(np.real(np.linalg.eigvals(P2))) < tol:
        raise ValueError("lpic2hpic: Upgrade impossible (P2 not positive definite).")

    Ki_new = -np.linalg.inv(P2) @ B.T @ P

    # --- Compute admissible mu range ---
    sqrtP = sqrtm(P)
    inv_sqrtP = np.linalg.inv(sqrtP)
    M_sym = sqrtP @ G0 @ inv_sqrtP + inv_sqrtP @ G0.T @ sqrtP
    lambdas = np.real(np.linalg.eigvals(M_sym))
    lambda_min = np.min(lambdas)
    lambda_max = np.max(lambdas)

    if lambda_max > tol:
        mu_min = max(-0.5, -1.0 / lambda_max + tol)
    else:
        mu_min = -0.5

    if lambda_min < -tol:
        mu_max = min(1.0 / k, -1.0 / lambda_min)
    else:
        mu_max = 1.0 / k

    return K0, G0, P, Ki_new, mu_min, mu_max
