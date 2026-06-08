"""
Upgrade Linear Proportional Control (LPC) to Homogeneous Proportional Control (HPC).

Original MATLAB: lpc2hpc.m

The upgrade:
    Linear:  u = K * x
    HPC:     u = K0 * x + nx^(1+mu) * (K - K0) * d(-ln(nx)) * x

where nx is the canonical homogeneous norm of x.
"""

import numpy as np
from scipy.linalg import sqrtm
from .block_con import block_con


def lpc2hpc(A, B, K):
    """
    Upgrade a linear proportional controller u = K*x to a Homogeneous
    Proportional Controller (HPC).

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    B : (n, m) ndarray
        Control matrix.
    K : (m, n) ndarray
        Linear feedback gain matrix.  A + B*K must be Hurwitz.

    Returns
    -------
    K0 : (m, n) ndarray
        Feedback gain of the linear component of HPC.
    G0 : (n, n) ndarray
        Matrix defining the generator of dilation: Gd = I + mu*G0.
    P : (n, n) ndarray
        Positive definite matrix defining the canonical homogeneous norm
        induced by sqrt(x'*P*x).
    mu_min : float
        Minimal admissible homogeneity degree (mu_min <= mu).
    mu_max : float
        Maximal admissible homogeneity degree (mu <= mu_max).
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
        raise ValueError("lpc2hpc: The system is not controllable.")

    if np.linalg.det(U_ctrb @ U_ctrb.T) < tol:
        print("Warning: The system is weakly controllable. "
              "Parameters may be badly tuned due to computation issues.")

    # --- Stability margin check ---
    rho = -max(np.real(np.linalg.eigvals(A + B @ K))) * 0.001
    if rho < tol:
        raise ValueError("lpc2hpc: The linear control system does not have "
                         "sufficient stability margin. Upgrade impossible.")

    # --- Special case: B is square and invertible ---
    if np.linalg.matrix_rank(B) == n:
        K0 = -B.T @ np.linalg.inv(B @ B.T) @ A
        K_out = -B.T @ np.linalg.inv(B @ B.T)
        G0 = np.eye(n)
        P = np.eye(n)
        # mu_min, mu_max can be arbitrary for fully actuated systems
        return K0, G0, P, -1.0, 1.0

    # --- Block decomposition ---
    T, nt = block_con(A, B)
    if isinstance(T, int) and T == 0:
        raise ValueError("lpc2hpc: Block decomposition failed.")

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

    # --- Compute P from Lyapunov equation ---
    # (A+B*K)'*P + P*(A+B*K) = -2*eye(n)
    Acl = A + B @ K
    I_n = np.eye(n)

    # Solve via Kronecker product:
    # kron(I, Acl') + kron(Acl', I)  *  vec(P) = -2*vec(I)
    W0 = np.kron(I_n, Acl.T) + np.kron(Acl.T, I_n)
    zet0 = -2.0 * I_n.flatten('F')
    v_P = np.linalg.solve(W0, zet0)
    P = v_P.reshape((n, n), order='F')

    # --- Compute admissible mu range ---
    sqrtP = sqrtm(P)
    inv_sqrtP = np.linalg.inv(sqrtP)
    M_sym = sqrtP @ G0 @ inv_sqrtP + inv_sqrtP @ G0.T @ sqrtP
    lambdas = np.real(np.linalg.eigvals(M_sym))
    lambda_min = np.min(lambdas)
    lambda_max = np.max(lambdas)

    if lambda_max > tol:
        mu_min = max(-1.0, -1.0 / lambda_max + tol)
    else:
        mu_min = -1.0

    if lambda_min < -tol:
        mu_max = min(1.0 / k, -1.0 / lambda_min)
    else:
        mu_max = 1.0 / k

    return K0, G0, P, mu_min, mu_max
