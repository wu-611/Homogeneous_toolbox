"""
Direct HPC design (without linear prototype).

Original MATLAB: hpc_design.m

Designs an HPC controller:
    u = K0*x + nx^(1+mu) * K * d(-ln(nx)) * x

that stabilizes dx/dt = A*x + B*u in:
  (a) finite time if mu < 0
  (b) nearly fixed time if mu > 0
"""

import numpy as np
from scipy.linalg import sqrtm
from .block_con import block_con


def hpc_design(A, B, mu=-1.0, rho=1.0, gamma=0.0, R=None):
    """
    Design HPC parameters directly (no linear prototype required).

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    B : (n, m) ndarray
        Control matrix.
    mu : float, optional
        Homogeneity degree (default -1, sliding-mode-like).
    rho : float, optional
        Convergence time tuning parameter (default 1).
    gamma : float, optional
        Matched perturbation bound (default 0).
    R : (n, n) ndarray, optional
        Shape matrix for mismatched perturbations (default B*B').

    Returns
    -------
    K0 : (m, n) ndarray
        Linear component gain.
    K : (m, n) ndarray
        Nonlinear component gain.
    Gd : (n, n) ndarray
        Generator of dilation.
    P : (n, n) ndarray
        Positive definite matrix for the canonical homogeneous norm.
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
        if np.linalg.matrix_rank(U_ctrb, tol) == n:
            k_ctrb = i
        tA = tA @ A

    if k_ctrb == 0:
        raise ValueError("hpc_design: The system is not controllable.")

    # --- Validate homogeneity degree ---
    if mu > 1.0 / k_ctrb:
        raise ValueError("hpc_design: Requested homogeneity degree is invalid. "
                         "Choose mu closer to 0.")
    if mu < -1.0:
        raise ValueError("hpc_design: Requested homogeneity degree is invalid. "
                         "Select mu closer to 0.")

    if rho <= 0:
        raise ValueError("hpc_design: Convergence time tuning parameter rho "
                         "must be positive.")

    if gamma < 0:
        raise ValueError("hpc_design: Perturbation bound gamma must be "
                         "non-negative.")

    if R is None:
        R_mat = B @ B.T
    else:
        if np.linalg.norm(R - R.T) > tol:
            raise ValueError("hpc_design: Perturbation shape matrix R must be "
                             "symmetric.")
        if np.min(np.real(np.linalg.eigvals(R))) < -tol:
            raise ValueError("hpc_design: R must be non-negative definite.")
        R_mat = R

    if np.linalg.det(U_ctrb @ U_ctrb.T) < tol:
        print("Warning: Weakly controllable system. Parameters may be badly tuned.")

    R_mat = gamma * R_mat

    # --- Special case: B square and invertible ---
    if np.linalg.matrix_rank(B) == n:
        K0 = -B.T @ np.linalg.inv(B @ B.T) @ A
        K = -B.T @ np.linalg.inv(B @ B.T)
        Gd = np.eye(n)
        P = np.eye(n)
        return K0, K, Gd, P

    # --- Block decomposition ---
    T, nt = block_con(A, B)
    if isinstance(T, int) and T == 0:
        raise ValueError("hpc_design: Block decomposition failed.")

    try:
        Anew = T @ A @ np.linalg.inv(T)
        Rt = T @ R_mat @ T.T
        k = len(nt)

        # Build block row start indices (0-based)
        n_ind = [0]
        s = 0
        for i in range(k - 1):
            s += nt[i]
            n_ind.append(s)

        # --- Build G0 and Gd ---
        vG0 = []
        for i in range(k):
            vG0.extend([mu * (k - 1 - i)] * nt[i])
        Gd = np.eye(n) - np.diag(vG0)
        Gd_tmp = Gd.copy()
        Gd = (rho + 0.5 * gamma) * Gd

        # --- Compute K0 ---
        Bnew = T @ B
        B0 = Bnew[n_ind[-1]:n, :m]
        K0 = -B0.T @ np.linalg.inv(B0 @ B0.T) @ Anew[n_ind[-1]:n, :n]
        A0 = Anew + Bnew @ K0
        K0 = K0 @ T

        # --- Compute P and K via block recursion ---
        Xii = np.eye(nt[0])
        X = Xii.copy()
        Gii = Gd[n_ind[0]:n_ind[0]+nt[0], n_ind[0]:n_ind[0]+nt[0]]
        Zii = Xii @ Gii.T + Gii @ Xii
        Rii = Rt[n_ind[0]:n_ind[0]+nt[0], n_ind[0]:n_ind[0]+nt[0]]
        Z = Zii.copy()

        for i in range(k - 1):
            # Ai_{i+1} block
            Ai_i1 = A0[n_ind[i]:n_ind[i]+nt[i],
                       n_ind[i+1]:n_ind[i+1]+nt[i+1]]
            Xi_i1 = -0.5 * (Zii + Rii) @ np.linalg.inv(Ai_i1 @ Ai_i1.T) @ Ai_i1
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
                Rj_i = Rt[n_ind[j]:n_ind[j]+nt[j],
                          n_ind[i]:n_ind[i]+nt[i]]
                term = -(Aj_j1 @ Xj1_i + Gjj @ Xj_i + Xj_i @ Gii.T + Rj_i)
                new_block = term @ np.linalg.inv(Ai_i1 @ Ai_i1.T) @ Ai_i1
                Xl = np.vstack([new_block, Xl])
                j -= 1

            # Compute new Gii for next iteration
            G_ = Gd[:n_ind[i+1], :n_ind[i+1]]
            Gii = Gd[n_ind[i+1]:n_ind[i+1]+nt[i+1],
                     n_ind[i+1]:n_ind[i+1]+nt[i+1]]
            Rii = Rt[n_ind[i+1]:n_ind[i+1]+nt[i+1],
                     n_ind[i+1]:n_ind[i+1]+nt[i+1]]

            Zl = Xl @ Gii.T + G_ @ Xl
            sg = np.min(np.real(np.linalg.eigvals(Gii.T + Gii)))
            lambda1 = max(np.real(np.linalg.eigvals(Xl.T @ np.linalg.inv(X) @ Xl))) + 1.0
            lambda2 = max(np.real(np.linalg.eigvals(Zl.T @ np.linalg.inv(Z) @ Zl))) / sg + 1.0
            lam = max(lambda1, lambda2)

            # Augment X and Z
            Xii = lam * np.eye(Xl.shape[1])
            X = np.block([[X, Xl], [Xl.T, Xii]])
            Zii = Xii @ Gii.T + Gii @ Xii
            Z = np.block([[Z, Zl], [Zl.T, Zii]])

        # --- Compute K ---
        tmpM = (A0 + Gd) @ X + X @ (A0 + Gd).T + Rt
        B0Y = np.hstack([
            -tmpM[n_ind[-1]:n_ind[-1]+nt[-1], :n_ind[-1]],
            -0.5 * tmpM[n_ind[-1]:n_ind[-1]+nt[-1], n_ind[-1]:n_ind[-1]+nt[-1]]
        ])
        Y = B0.T @ np.linalg.inv(B0 @ B0.T) @ B0Y
        K = Y @ np.linalg.inv(X)

        Gd = np.linalg.inv(T) @ Gd_tmp @ T
        X = np.linalg.inv(T) @ X @ np.linalg.inv(T).T
        P = np.linalg.inv(X)
        K = K @ T

    except Exception:
        raise ValueError("hpc_design: Control gains cannot be found with "
                         "block decomposition method.")

    return K0, K, Gd, P
