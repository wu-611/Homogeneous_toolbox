"""
Direct FHPC design (Fixed-time Homogeneous Proportional Control).

Original MATLAB: fhpc_design.m

    FHPC uses two homogeneity degrees mu1 (negative) and mu2 (positive)
    to achieve fixed-time convergence:
        |x(t)| = 0  for all t >= Tmax = 1/(-mu1*rho) + 1/(mu2*rho)
"""

import numpy as np
from scipy.linalg import sqrtm
from .block_con import block_con


def fhpc_design(A, B):
    """
    Design FHPC parameters for fixed-time stabilization.

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    B : (n, m) ndarray
        Control matrix.

    Returns
    -------
    K0 : (m, n) ndarray
        Linear component gain.
    K : (m, n) ndarray
        Nonlinear component gain.
    G0 : (n, n) ndarray
        Matrix defining generators: Gd1 = I + mu1*G0, Gd2 = I + mu2*G0.
    P : (n, n) ndarray
        Positive definite matrix for homogeneous norm.
    mu1 : float
        Minimal admissible negative homogeneity degree.
    mu2 : float
        Maximal admissible positive homogeneity degree.
    rho : float
        Convergence tuning parameter.
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
        raise ValueError("fhpc_design: The system is not controllable.")

    if np.linalg.det(U_ctrb @ U_ctrb.T) < tol:
        print("Warning: Weakly controllable system. Parameters may be badly tuned.")

    # --- Special case: B square ---
    if np.linalg.matrix_rank(B) == n:
        K0 = -B.T @ np.linalg.inv(B @ B.T) @ A
        K = -B.T @ np.linalg.inv(B @ B.T)
        G0 = np.eye(n)
        P = np.eye(n)
        mu1 = -1.0
        mu2 = 1.0
        rho = 2.0
        return K0, K, G0, P, mu1, mu2, rho

    # --- Block decomposition ---
    T, nt = block_con(A, B)
    if isinstance(T, int) and T == 0:
        raise ValueError("fhpc_design: Block decomposition failed.")

    try:
        Anew = T @ A @ np.linalg.inv(T)
        k = len(nt)

        # Build block row start indices (0-based)
        n_ind = [0]
        s = 0
        for i in range(k - 1):
            s += nt[i]
            n_ind.append(s)

        # --- G0 (negative diagonal) ---
        vG0 = []
        for i in range(k):
            vG0.extend([(k - 1 - i)] * nt[i])
        G0 = -np.diag(vG0)
        Gd = np.eye(n)   # temporary (mu=0 for X computation)
        R_mat = np.eye(n)

        # --- K0 ---
        Bnew = T @ B
        B0 = Bnew[n_ind[-1]:n, :m]
        K0 = -B0.T @ np.linalg.inv(B0 @ B0.T) @ Anew[n_ind[-1]:n, :n]
        A0 = Anew + Bnew @ K0
        K0 = K0 @ T

        # --- Compute X and P (same recursion as hpc_design with Gd=I) ---
        Xii = np.eye(nt[0])
        X = Xii.copy()
        Gii = Gd[n_ind[0]:n_ind[0]+nt[0], n_ind[0]:n_ind[0]+nt[0]]
        Zii = Xii @ Gii.T + Gii @ Xii
        R_ii = R_mat[n_ind[0]:n_ind[0]+nt[0], n_ind[0]:n_ind[0]+nt[0]]
        Z = Zii.copy()

        for i in range(k - 1):
            Ai_i1 = A0[n_ind[i]:n_ind[i]+nt[i],
                       n_ind[i+1]:n_ind[i+1]+nt[i+1]]
            Xi_i1 = -0.5 * (Zii + R_ii) @ np.linalg.inv(Ai_i1 @ Ai_i1.T) @ Ai_i1
            Xl = Xi_i1.copy()

            j = i - 1
            while j >= 0:
                Aj_j1 = A0[n_ind[j]:n_ind[j]+nt[j],
                           n_ind[j+1]:n_ind[j+1]+nt[j+1]]
                Gjj = Gd[n_ind[j]:n_ind[j]+nt[j],
                         n_ind[j]:n_ind[j]+nt[j]]
                Xj1_i = X[n_ind[j+1]:n_ind[j+1]+nt[j+1],
                          n_ind[i]:n_ind[i]+nt[i]]
                Xj_i = X[n_ind[j]:n_ind[j]+nt[j],
                         n_ind[i]:n_ind[i]+nt[i]]
                Rj_i = R_mat[n_ind[j]:n_ind[j]+nt[j],
                             n_ind[i]:n_ind[i]+nt[i]]
                term = -(Aj_j1 @ Xj1_i + Gjj @ Xj_i + Xj_i @ Gii.T + Rj_i)
                new_block = term @ np.linalg.inv(Ai_i1 @ Ai_i1.T) @ Ai_i1
                Xl = np.vstack([new_block, Xl])
                j -= 1

            G_ = Gd[:n_ind[i+1], :n_ind[i+1]]
            Gii = Gd[n_ind[i+1]:n_ind[i+1]+nt[i+1],
                     n_ind[i+1]:n_ind[i+1]+nt[i+1]]
            R_ii = R_mat[n_ind[i+1]:n_ind[i+1]+nt[i+1],
                         n_ind[i+1]:n_ind[i+1]+nt[i+1]]

            Zl = Xl @ Gii.T + G_ @ Xl
            sg = np.min(np.real(np.linalg.eigvals(Gii.T + Gii)))
            lambda1 = max(np.real(np.linalg.eigvals(Xl.T @ np.linalg.inv(X) @ Xl))) + 1.0
            lambda2 = max(np.real(np.linalg.eigvals(Zl.T @ np.linalg.inv(Z) @ Zl))) / sg + 1.0
            lam = max(lambda1, lambda2)

            Xii = lam * np.eye(Xl.shape[1])
            X = np.block([[X, Xl], [Xl.T, Xii]])
            Zii = Xii @ Gii.T + Gii @ Xii
            Z = np.block([[Z, Zl], [Zl.T, Zii]])

        # --- Compute K ---
        tmpM = (A0 + Gd) @ X + X @ (A0 + Gd).T + R_mat
        B0Y = np.hstack([
            -tmpM[n_ind[-1]:n_ind[-1]+nt[-1], :n_ind[-1]],
            -0.5 * tmpM[n_ind[-1]:n_ind[-1]+nt[-1], n_ind[-1]:n_ind[-1]+nt[-1]]
        ])
        Y = B0.T @ np.linalg.inv(B0 @ B0.T) @ B0Y
        K = Y @ np.linalg.inv(X)

        X = np.linalg.inv(T) @ X @ np.linalg.inv(T).T
        P = np.linalg.inv(X)
        K = K @ T
        G0_full = np.linalg.inv(T) @ G0 @ T

        # --- Compute admissible mu1, mu2 ---
        sqrtP = sqrtm(P)
        inv_sqrtP = np.linalg.inv(sqrtP)
        M_sym = sqrtP @ G0_full @ inv_sqrtP + inv_sqrtP @ G0_full.T @ sqrtP
        lambdas = np.real(np.linalg.eigvals(M_sym))
        lambda_min = np.min(lambdas)
        lambda_max = np.max(lambdas)

        if lambda_max > tol:
            mu1 = max(-1.0, -1.0 / lambda_max + tol)
        else:
            mu1 = -1.0

        if lambda_min < -tol:
            mu2 = min(1.0 / k, -1.0 / lambda_min - tol)
        else:
            mu2 = 1.0 / k

        # --- Compute rho ---
        delta = max(mu2 * lambda_max, mu1 * lambda_min)
        if delta < 0:
            rho = 1.0
        else:
            rho = (2.0 + np.min(np.real(np.linalg.eigvals(X)))) / (2.0 + delta)

    except Exception:
        raise ValueError("fhpc_design: Control gains cannot be found with "
                         "block decomposition method.")

    return K0, K, G0_full, P, mu1, mu2, rho
