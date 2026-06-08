"""
Orthogonal transformation to block controllability staircase form.

Original MATLAB: trans_con.m
"""

import numpy as np
from scipy.linalg import null_space


def trans_con(A, B, tol=1e-6):
    """
    Compute an orthogonal matrix T that transforms a controllable pair {A, B}
    to the canonical block controllability staircase form:

        T*A*T' = [ A11  A12   0  ...   0   ;
                    A21  A22  A23 ...   0   ;
                    ...  ...  ... ...  ...  ;
                     *    *    *  ... Ak-1k ;
                    Ak1  Ak2  Ak3 ...  Akk  ]

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
        Rank tolerance (default 1e-6).

    Returns
    -------
    T : (n, n) ndarray or 0
        Orthogonal transformation matrix. 0 if system is not controllable.
    nt : (k,) ndarray
        List of block sizes [n1, ..., nk].
    """
    n = A.shape[0]

    # --- Controllability check ---
    U = np.zeros((n, 0))
    Ak = np.eye(n)
    for i in range(n):
        U = np.hstack([U, Ak @ B])
        Ak = Ak @ A

    if np.linalg.matrix_rank(U, tol) < n:
        print("The pair {A,B} is not controllable.")
        return 0, np.array([])

    # --- Staircase orthogonal transformation ---
    T = np.eye(n)
    Ak_mat = A.copy()
    Bk = B.copy()
    l = 0
    nt_list = []

    while np.linalg.matrix_rank(Bk, tol) < Ak_mat.shape[0]:
        rk = np.linalg.matrix_rank(Bk, tol)
        nt_list.insert(0, rk)

        # Orthogonal complement of Bk (rows of B_ort span the nullspace of Bk')
        # null_space(Bk.T) gives orthonormal basis for nullspace of Bk'
        B_ort = null_space(Bk.T).T   # shape (nk - rk, nk)
        # Orthogonal complement of B_ort (rows span the rowspace of Bk')
        B_p = null_space(B_ort).T    # shape (rk, nk)

        if Ak_mat.shape[0] < n:
            # MATLAB: [[B_ort; B_p] zeros(nk,l); zeros(l,nk) eye(l)]
            # B_ort (nk-rk, nk) stacked vertically with B_p (rk, nk) → (nk, nk)
            top_left = np.vstack([B_ort, B_p])
            top_row = np.hstack([top_left, np.zeros((Ak_mat.shape[0], l))])
            bottom_row = np.hstack([np.zeros((l, Ak_mat.shape[0])), np.eye(l)])
            T = np.vstack([top_row, bottom_row]) @ T
        else:
            T = np.vstack([B_ort, B_p])

        l += rk
        Bk = B_ort @ Ak_mat @ B_p.T
        Ak_mat = B_ort @ Ak_mat @ B_ort.T

    # The final block is the last Bk
    nt_list.insert(0, np.linalg.matrix_rank(Bk, tol))
    nt = np.array(nt_list, dtype=int)

    return T, nt
