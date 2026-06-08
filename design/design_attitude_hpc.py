"""
Attitude loop so(3) homogeneous controller design.

Based on Zhou et al. (2023) "Generalized Homogeneous Rigid-Body Attitude Control".

State: xi = [theta_e; omega_e] in R^6
  theta_e: exponential coordinate attitude error (so(3) ~ R^3)
  omega_e: angular velocity error in body frame

Key design parameters:
  K1  : proportional gain matrix (3x3 positive definite)
  k2  : damping gain (scalar)
  mu  : homogeneity degree (mu<0: finite-time, mu=0: exponential)
  eps : coupling parameter in Lyapunov matrix P
"""

import numpy as np
from scipy.linalg import expm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hcs_toolbox_py import hnorm


class AttitudeHPC:
    """
    Attitude-loop Homogeneous Controller on so(3) (exponential coordinates).

    Control law:
        u_hom = ||xi||_d^(1+mu) * K * expm(-ln||xi||_d * Gd) * xi

    where:
        xi = [theta_e; omega_e]
        K = [-K1, -k2*I3]
        Gd = block_diag((1-mu)*I3, I3)
    """

    def __init__(self, J, K1=None, k2=None, mu=0.0, eps=None):
        """
        Parameters
        ----------
        J : (3, 3) ndarray
            Inertia matrix.
        K1 : (3, 3) ndarray or float
            Proportional attitude gain. If float, K1 = K1_val * I3.
            Default: 12 * I3.
        k2 : float
            Damping gain. Default: 6.
        mu : float
            Homogeneity degree.
            mu = 0: exponential convergence (classical geometric PD)
            mu < 0: finite-time convergence
        eps : float or None
            Coupling parameter in P matrix. If None, auto-computed.
        """
        self.J = np.asarray(J)
        self.J_inv = np.linalg.inv(self.J)
        self.mu = mu
        self.n = 6

        # Gains
        # IMPORTANT: u_hom is angular acceleration [rad/s^2].
        # Torque M = J * u_hom (+ Coriolis compensation).
        # So effective torque gain ≈ J * K1.
        if K1 is None:
            self.K1 = 200.0 * np.eye(3)
        elif np.isscalar(K1):
            self.K1 = K1 * np.eye(3)
        else:
            self.K1 = np.asarray(K1)

        if k2 is None:
            self.k2 = 100.0
        else:
            self.k2 = k2

        # Full gain matrix K = [-K1, -k2*I3]
        self.K = np.hstack([-self.K1, -self.k2 * np.eye(3)])

        # Dilation generator (from Zhou 2023, Eq. 21a)
        self.Gd = np.zeros((6, 6))
        self.Gd[0:3, 0:3] = (1 - mu) * np.eye(3)
        self.Gd[3:6, 3:6] = np.eye(3)

        # Shape matrix P (from Zhou 2023, Eq. 21b)
        if eps is None:
            eps = self._compute_eps()
        self.eps = eps

        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = np.eye(3)
        self.P[0:3, 3:6] = eps * np.eye(3)
        self.P[3:6, 0:3] = eps * np.eye(3)
        self.P[3:6, 3:6] = np.linalg.inv(self.K1)

        # Verify P > 0 and PGd + Gd'P > 0
        self._validate()

    def _compute_eps(self):
        """Compute admissible epsilon satisfying P>0 and PGd+Gd'P>0."""
        # Constraint 1: P > 0 requires eps^2 < lambda_min(K1^{-1})
        K1_inv = np.linalg.inv(self.K1)
        eps_pd = np.sqrt(np.min(np.linalg.eigvals(K1_inv)))

        # Constraint 2: PGd + Gd'P > 0
        # For the block structure, this gives an upper bound on eps
        # eps < 2*sqrt(1-mu) / ((2-mu)*sqrt(lambda_max(K1)))
        lambda_max = np.max(np.linalg.eigvals(self.K1))
        if abs(self.mu - 1.0) < 1e-10:
            eps_gd = np.inf
        else:
            eps_gd = (2.0 * np.sqrt(1.0 - self.mu) /
                      ((2.0 - self.mu) * np.sqrt(lambda_max)))

        eps = 0.5 * min(eps_pd, eps_gd)
        return max(eps, 1e-6)

    def _validate(self):
        """Check that P > 0 and PGd + Gd'P > 0."""
        eig_P = np.linalg.eigvals(self.P)
        if np.any(eig_P <= 0):
            print(f"Warning: P has non-positive eigenvalues: {eig_P}")

        M = self.P @ self.Gd + self.Gd.T @ self.P
        eig_M = np.linalg.eigvals(M)
        if np.any(eig_M <= 0):
            print(f"Warning: PGd+Gd'P has non-positive eigenvalues: {eig_M}")

    def homogeneous_norm(self, xi):
        """Compute canonical homogeneous norm of xi."""
        return hnorm(xi, self.Gd, self.P)

    def compute_virtual_control(self, theta_e, omega_e):
        """
        Compute virtual control u_hom (angular acceleration error).

        Parameters
        ----------
        theta_e : (3,) ndarray
            Exponential coordinate attitude error.
        omega_e : (3,) ndarray
            Angular velocity error.

        Returns
        -------
        u_hom : (3,) ndarray
            Homogeneous control in so(3) coordinates.
        """
        xi = np.concatenate([theta_e, omega_e])

        if np.linalg.norm(xi) < 1e-12:
            return np.zeros(3)

        nx = self.homogeneous_norm(xi)

        alpha = 0.1
        beta = 10.0
        nx_sat = max(alpha, min(beta, nx))

        if nx_sat < 1e-20:
            return np.zeros(3)

        # u_hom = nx^(1+mu) * K * expm(-ln(nx) * Gd) * xi
        d_proj = expm(-np.log(nx_sat) * self.Gd) @ xi
        u_hom = (nx_sat ** (1.0 + self.mu)) * (self.K @ d_proj)
        return u_hom

    def compute_torque(self, theta_e, omega_e, R_d, omega, omega_d, omega_d_dot):
        """
        Compute actual body torque from virtual control.

        Implements the moment mapping (Zhou 2023, Eq. 23):
            M = J * (R_d' * u_hom - omega_d^ x omega + omega_d_dot) + omega x J*omega

        Parameters
        ----------
        theta_e : (3,) ndarray
            Attitude error in exponential coordinates.
        omega_e : (3,) ndarray
            Angular velocity error.
        R_d : (3,3) ndarray
            Desired attitude.
        omega : (3,) ndarray
            Current body angular velocity.
        omega_d : (3,) ndarray
            Desired body angular velocity.
        omega_d_dot : (3,) ndarray
            Desired body angular acceleration.

        Returns
        -------
        M : (3,) ndarray
            Body torque command.
        """
        u_hom = self.compute_virtual_control(theta_e, omega_e)

        # M = J * (R_d' * u_hom - omega_d^ x omega + omega_d_dot) + omega x J*omega
        from models.quadrotor_se3 import hat
        M = (self.J @ (R_d.T @ u_hom - np.cross(omega_d, omega) + omega_d_dot) +
             np.cross(omega, self.J @ omega))
        return M
