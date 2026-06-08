"""
Position loop HPC design for quadrotor translational dynamics.

Each channel (x, y, z) is a double integrator:
    A_i = [[0, 1], [0, 0]],  B_i = [[0], [1/m]]

The upgrade uses lpc2hpc to obtain K0_i, G0_i, P_i from linear gains.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hcs_toolbox_py import lpc2hpc, hnorm


class PositionHPC:
    """
    Position-loop Homogeneous Proportional Controller for quadrotor.

    Three independent channels (x, y, z), each a double integrator.
    Output: desired acceleration (F_des / m) in the inertial frame.
    """

    def __init__(self, m, K_linear=None, mu=0.0):
        """
        Parameters
        ----------
        m : float
            Vehicle mass [kg].
        K_linear : (3, 6) ndarray or None
            Linear feedback gains for [e_x, e_y, e_z, e_vx, e_vy, e_vz].
            If None, uses default pole-placement gains.
        mu : float
            Homogeneity degree for position loop.
            mu=0: exponential convergence (uniform dilation)
            mu<0: finite-time convergence
        """
        self.m = m
        self.mu = mu
        self.g = 9.81

        # Double integrator for each channel
        self.A = np.array([[0., 1.], [0., 0.]])
        self.B = np.array([[0.], [1. / m]])

        if K_linear is None:
            # Default: poles at [-2, -3] for each channel
            # For double integrator: K = [k_p, k_d] with u = -k_p*e - k_d*e_v
            # Closed loop: s^2 + (k_d/m)*s + (k_p/m) = 0
            # Desired: (s+2)(s+3) = s^2 + 5s + 6
            self.K_linear = np.array([-6.0 * m, -5.0 * m])

        # Upgrade to homogeneous controller
        self._design()

    def _design(self):
        """Run lpc2hpc to obtain homogeneous parameters."""
        K0, G0, P, mu_min, mu_max = lpc2hpc(self.A, self.B,
                                             self.K_linear.reshape(1, -1))

        self.K0 = K0
        self.G0 = G0
        self.P = P
        self.mu_min = mu_min
        self.mu_max = mu_max

        # Verify mu is in admissible range
        mu = np.clip(self.mu, self.mu_min + 1e-6, self.mu_max - 1e-6)
        if abs(mu - self.mu) > 1e-10:
            print(f"Warning: mu={self.mu} outside [{self.mu_min:.4f}, "
                  f"{self.mu_max:.4f}], clipped to {mu:.4f}")
        self.mu = mu

        # Build dilation generator
        if abs(self.mu) < 1e-10:
            self.Gd = np.eye(2)
        else:
            self.Gd = np.eye(2) + self.mu * self.G0

        # Nonlinear gain
        self.K_nl = self.K_linear.reshape(1, -1) - self.K0

        # Homogeneous norm function
        self._hn_fun = lambda x: hnorm(x, self.Gd, self.P)

    def compute_control(self, e_pos, e_vel):
        """
        Compute desired acceleration for a single channel.

        Parameters
        ----------
        e_pos : float
            Position error (scalar).
        e_vel : float
            Velocity error (scalar).

        Returns
        -------
        u_pos : float
            Desired acceleration component.
        """
        from hcs_toolbox_py import e_hpc

        x = np.array([e_pos, e_vel])
        u, = e_hpc(x, self.K0, self.K_nl, self.Gd, self.mu, self._hn_fun,
                   alpha=0.1, beta=10.0)
        return u

    def compute_control_vector(self, e_pos_vec, e_vel_vec):
        """
        Compute desired acceleration for all 3 channels.

        Parameters
        ----------
        e_pos_vec : (3,) ndarray
            Position errors [e_x, e_y, e_z].
        e_vel_vec : (3,) ndarray
            Velocity errors [e_vx, e_vy, e_vz].

        Returns
        -------
        u_pos : (3,) ndarray
            Desired acceleration vector (in inertial frame, Z-up).
        """
        return np.array([
            self.compute_control(e_pos_vec[0], e_vel_vec[0]),
            self.compute_control(e_pos_vec[1], e_vel_vec[1]),
            self.compute_control(e_pos_vec[2], e_vel_vec[2])
        ])

    def get_desired_thrust_direction(self, e_pos_vec, e_vel_vec):
        """
        Compute desired thrust direction from position controller output.

        Parameters
        ----------
        e_pos_vec, e_vel_vec : (3,) ndarray

        Returns
        -------
        F_des : (3,) ndarray
            Total desired force vector.
        b3_des : (3,) ndarray
            Desired thrust direction (unit vector in body Z).
        thrust_magnitude : float
            Magnitude of desired thrust.
        """
        u_pos = self.compute_control_vector(e_pos_vec, e_vel_vec)
        # Add gravity compensation
        F_des = u_pos + np.array([0., 0., self.g])
        thrust_magnitude = np.linalg.norm(F_des)

        if thrust_magnitude < 1e-10:
            b3_des = np.array([0., 0., 1.])
        else:
            b3_des = F_des / thrust_magnitude

        return F_des, b3_des, thrust_magnitude
