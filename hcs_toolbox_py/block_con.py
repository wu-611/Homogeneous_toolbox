"""
Block controllability canonical form.

Original MATLAB: block_con.m
"""

import numpy as np
from .trans_con import trans_con


def block_con(A, B, tol=1e-6):
    """
    Compute an invertible matrix T that transforms a controllable pair {A, B}
    to the canonical block controllability form:

        T*A*inv(T) = [ 0   A12  0  ...   0   ;
                        0   0   A23 ...   0   ;
                       ... ... ... ...  ...  ;
                        0   0   0  ...  Ak-1k ;
                       Ak1 Ak2 Ak3 ...  Akk  ]

    and

        T*B = [0; B0],

    where Aij is a block of size (ni x nj) and B0 is full row rank (nk x m).

    Parameters
    ----------
    A : (n, n) ndarray
        System matrix.
    B : (n, m) ndarray
        Control matrix.
    tol : float, optional
        Rank tolerance.

    Returns
    -------
    T : (n, n) ndarray or 0
        Transformation matrix. 0 if decomposition fails.
    nt : (k,) ndarray
        Block sizes [n1, ..., nk].
    """
    # Step 1: Orthogonal transformation to staircase form
    T_ort, nt = trans_con(A, B, tol)
    if isinstance(T_ort, int) and T_ort == 0:
        return 0, np.array([])

    k = len(nt)
    n = A.shape[0]

    # Build block row start indices (0-based)
    ni = [0]
    j = 0
    for i in range(k - 1):
        j += nt[i]
        ni.append(j)

    A_curr = T_ort @ A @ T_ort.T
    U_mat = T_ort.copy()

    # Step 2: Triangular transformation to zero out lower-left blocks
    Phi = np.eye(n)
    for i in range(k - 1):
        # Extract the super-diagonal block Ai_{i+1}
        r_start = ni[i]
        r_end = ni[i] + nt[i]
        c_start = ni[i] + nt[i]
        c_end = ni[i] + nt[i] + nt[i+1]
        temp_A = A_curr[r_start:r_end, c_start:c_end]

        # Build the row block that will zero out lower elements
        inv_part = np.linalg.inv(temp_A @ temp_A.T) @ temp_A
        left_part = -inv_part @ A_curr[r_start:r_end, :c_start]
        right_part = np.eye(nt[i+1])
        zero_part = np.zeros((nt[i+1], n - c_start - nt[i+1]))

        temp_block = np.hstack([left_part, right_part, zero_part])
        temp_T = np.eye(n)
        temp_T[ni[i+1]:ni[i+1]+nt[i+1], :] = temp_block

        Phi = temp_T @ Phi
        A_curr = temp_T @ A_curr @ np.linalg.inv(temp_T)

    T = Phi @ T_ort
    return T, nt
