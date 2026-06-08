"""
Quadrotor SE(3) full dynamics model.

State: [x, y, z, vx, vy, vz, R(3x3), omega_x, omega_y, omega_z]
Input: [F_total, tau_x, tau_y, tau_z]  (thrust + body torques)

Based on Lee et al. (2010) geometric tracking control formulation.
"""

import numpy as np
from scipy.linalg import expm


def hat(v):
    """Skew-symmetric matrix from 3-vector: v^ (so(3) element)."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def vee(S):
    """Inverse of hat: extract 3-vector from skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class QuadrotorSE3:
    """
    Quadrotor dynamics on SE(3).

    State representation (18 elements):
        pos[0:3]   — position in inertial frame (Z-up)
        vel[3:6]   — velocity in inertial frame
        R[6:15]    — rotation matrix (3x3, stored flat, column-major)
        omega[15:18] — body angular velocity

    Control input (4 elements):
        thrust    — total thrust in body Z direction
        tau[1:3]  — body torques
    """

    def __init__(self, m=1.4, J=None, g=9.81):
        """
        Parameters
        ----------
        m : float
            Mass [kg].
        J : (3, 3) ndarray
            Inertia matrix [kg*m^2]. Default: diagonal [0.0211, 0.0219, 0.0366].
        g : float
            Gravitational acceleration [m/s^2].
        """
        self.m = m
        if J is None:
            self.J = np.diag([0.0211, 0.0219, 0.0366])
        else:
            self.J = np.asarray(J)
        self.g = g
        self.J_inv = np.linalg.inv(self.J)

    def dynamics(self, state, u):
        """
        Continuous-time SE(3) dynamics.

        Parameters
        ----------
        state : (18,) ndarray
            [pos(3), vel(3), R(9, flat), omega(3)]
        u : (4,) ndarray
            [thrust, tau_x, tau_y, tau_z]

        Returns
        -------
        dstate : (18,) ndarray
            Time derivative of state.
        """
        pos = state[0:3]
        vel = state[3:6]
        R = state[6:15].reshape(3, 3)
        omega = state[15:18]
        thrust = u[0]
        tau = u[1:4]

        # Position kinematics
        dpos = vel

        # Velocity dynamics (Newton)
        e3 = np.array([0., 0., 1.])
        dvel = np.array([0., 0., -self.g]) + (thrust / self.m) * (R @ e3)

        # Attitude kinematics (Poisson equation)
        dR = R @ hat(omega)

        # Angular velocity dynamics (Euler)
        domega = self.J_inv @ (tau - np.cross(omega, self.J @ omega))

        dstate = np.zeros(18)
        dstate[0:3] = dpos
        dstate[3:6] = dvel
        dstate[6:15] = dR.flatten()
        dstate[15:18] = domega
        return dstate

    def step_rk4(self, state, u, dt):
        """RK4 integration of SE(3) dynamics."""
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * dt * k1, u)
        k3 = self.dynamics(state + 0.5 * dt * k2, u)
        k4 = self.dynamics(state + dt * k3, u)
        state_new = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Re-orthogonalize rotation matrix (SVD projection onto SO(3))
        R = state_new[6:15].reshape(3, 3)
        U, _, Vt = np.linalg.svd(R)
        R_proj = U @ Vt
        if np.linalg.det(R_proj) < 0:
            R_proj = U @ np.diag([1, 1, -1]) @ Vt
        state_new[6:15] = R_proj.flatten()
        return state_new

    def make_state(self, pos, vel, R, omega):
        """Pack components into state vector."""
        state = np.zeros(18)
        state[0:3] = np.asarray(pos).flatten()
        state[3:6] = np.asarray(vel).flatten()
        state[6:15] = np.asarray(R).flatten()
        state[15:18] = np.asarray(omega).flatten()
        return state

    def unpack_state(self, state):
        """Extract components from state vector."""
        pos = state[0:3].copy()
        vel = state[3:6].copy()
        R = state[6:15].copy().reshape(3, 3)
        omega = state[15:18].copy()
        return pos, vel, R, omega


def exp_so3(theta):
    """Exponential map from so(3) to SO(3) (Rodrigues formula)."""
    theta = np.asarray(theta).flatten()
    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return np.eye(3)
    axis = theta / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def log_so3(R):
    """Logarithmic map from SO(3) to so(3) (exponential coordinates)."""
    R = np.asarray(R).reshape(3, 3)
    # Clamp trace to valid range for arccos
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    angle = np.arccos(tr)

    if angle < 1e-12:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-10:
        # 180-degree case: extract axis from R - I
        v = vee(R - np.eye(3))
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            return np.pi * v / v_norm
        return np.zeros(3)

    return (angle / (2 * np.sin(angle))) * vee(R - R.T)


def jacobian_r_inv(theta):
    """
    Inverse of right Jacobian of SO(3).

    J_r^{-1}(theta) relates angular velocity to exponential coordinate rate:
        d/dt theta = J_r^{-1}(theta) * omega
    """
    theta = np.asarray(theta).flatten()
    angle = np.linalg.norm(theta)

    if angle < 1e-12:
        return np.eye(3)

    K = hat(theta)
    a = 1 / (angle ** 2)
    b = (1 + np.cos(angle)) / (2 * angle * np.sin(angle))
    return np.eye(3) + 0.5 * K + (a - b) * K @ K
