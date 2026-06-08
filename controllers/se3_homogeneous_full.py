"""
SE(3) Geometric Homogeneous Controller — Full-State Feedback.

Combines:
  - Position HPC (3 independent double-integrator channels)
  - Attitude Homogeneous Controller on so(3) (Zhou 2023)
  - Attitude command computation (Lee et al. 2010)
  - Torque/thrust mapping

Architecture:
  pos_error → Position HPC → F_des → attitude_command → R_d, omega_d, omega_d_dot
  attitude_error → Attitude HPC → u_hom → torque_mapping → M

Supports:
  - mu_p = 0, mu_a = 0 : exponential convergence (Phase 1)
  - mu_p < 0, mu_a < 0 : finite-time convergence (Phase 2)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from design.design_position_hpc import PositionHPC
from design.design_attitude_hpc import AttitudeHPC
from design.attitude_command import (compute_desired_attitude,
                                      compute_attitude_error,
                                      compute_omega_error)
from design.torque_mapping import map_virtual_to_torque, map_accel_to_thrust
from models.quadrotor_se3 import hat


class SE3HomogeneousController:
    """
    SE(3) geometric homogeneous tracking controller.

    This controller replaces the linear PD terms in Lee et al. (2010)'s
    geometric controller with homogeneous controllers for both the
    position and attitude loops.
    """

    def __init__(self, m, J, g=9.81, mu_p=0.0, mu_a=0.0,
                 K_pos=None, K1_att=None, k2_att=None):
        """
        Parameters
        ----------
        m : float
            Vehicle mass [kg].
        J : (3,3) ndarray
            Inertia matrix.
        g : float
            Gravitational acceleration [m/s^2].
        mu_p : float
            Position loop homogeneity degree (default 0: exponential).
        mu_a : float
            Attitude loop homogeneity degree (default 0: exponential).
        K_pos : (3,6) ndarray or None
            Linear position gains. Default: pole placement at [-3, -4].
        K1_att : float or (3,3) ndarray
            Attitude proportional gain. Default: 12*I3.
        k2_att : float
            Attitude damping gain. Default: 6.
        """
        self.m = m
        self.J = np.asarray(J)
        self.g = g
        self.mu_p = mu_p
        self.mu_a = mu_a

        # Design position loop HPC
        self.pos_hpc = PositionHPC(m, K_pos, mu_p)

        # Design attitude loop HPC
        self.att_hpc = AttitudeHPC(J, K1_att, k2_att, mu_a)

    def compute_control(self, state, pos_d, vel_d, yaw_d, omega_d=None,
                        omega_d_dot=None, torque_limit=20.0, thrust_limits=(0.1, 80.0)):
        """
        Compute full control input [thrust, tau_x, tau_y, tau_z].

        Parameters
        ----------
        state : (18,) ndarray
            Current SE(3) state [pos, vel, R(flat), omega].
        pos_d : (3,) ndarray
            Desired position.
        vel_d : (3,) ndarray
            Desired velocity.
        yaw_d : float
            Desired yaw angle.
        omega_d : (3,) ndarray or None
            Desired angular velocity. Default: zeros.
        omega_d_dot : (3,) ndarray or None
            Desired angular acceleration. Default: zeros.
        torque_limit : float
            Max torque magnitude per axis [N·m].
        thrust_limits : (float, float)
            (min_thrust, max_thrust) [N].

        Returns
        -------
        u : (4,) ndarray
            [thrust, tau_x, tau_y, tau_z].
        """
        from models.quadrotor_se3 import QuadrotorSE3
        model = QuadrotorSE3(self.m, self.J, self.g)
        pos, vel, R, omega = model.unpack_state(state)

        if omega_d is None:
            omega_d = np.zeros(3)
        if omega_d_dot is None:
            omega_d_dot = np.zeros(3)

        # --- Position loop ---
        e_pos = pos - pos_d
        e_vel = vel - vel_d
        u_pos = self.pos_hpc.compute_control_vector(e_pos, e_vel)

        # Gravity-compensated desired force direction
        F_des = u_pos + np.array([0., 0., self.g])
        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)

        # --- Attitude command ---
        R_d = compute_desired_attitude(b3_des, yaw_d)

        # --- Attitude loop ---
        theta_e = compute_attitude_error(R, R_d)
        omega_e = compute_omega_error(omega, omega_d, R, R_d)
        u_hom = self.att_hpc.compute_virtual_control(theta_e, omega_e)

        # --- Torque mapping ---
        M = map_virtual_to_torque(u_hom, self.J, R_d, omega, omega_d, omega_d_dot)

        # --- Thrust (project desired force onto body Z axis) ---
        # u_pos is desired acceleration [m/s^2]
        # F_des_specific = u_pos + g*e3  is specific force [N/kg]
        # F_des = m * F_des_specific  is total desired force [N]
        # Thrust f along body Z produces force f * R*e3 in inertial frame
        # For force equilibrium: f * R*e3 ≈ F_des
        # → f = F_des · (R * e3)
        F_des_force = self.m * (u_pos + np.array([0., 0., self.g]))
        thrust = np.dot(F_des_force, R @ np.array([0., 0., 1.]))

        # Apply saturation
        thrust = np.clip(thrust, thrust_limits[0], thrust_limits[1])
        M = np.clip(M, -torque_limit, torque_limit)

        return np.array([thrust, M[0], M[1], M[2]])

    def get_debug_info(self, state, pos_d, vel_d, yaw_d):
        """Return intermediate signals for debugging/logging."""
        from models.quadrotor_se3 import QuadrotorSE3
        model = QuadrotorSE3(self.m, self.J, self.g)
        pos, vel, R, omega = model.unpack_state(state)

        e_pos = pos - pos_d
        e_vel = vel - vel_d
        u_pos = self.pos_hpc.compute_control_vector(e_pos, e_vel)

        F_des = u_pos + np.array([0., 0., self.g])
        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)

        R_d = compute_desired_attitude(b3_des, yaw_d)
        theta_e = compute_attitude_error(R, R_d)
        omega_e = R.T @ R_d @ (omega - np.zeros(3))  # simplified

        return {
            'e_pos': e_pos,
            'e_vel': e_vel,
            'u_pos': u_pos,
            'F_des': F_des,
            'b3_des': b3_des,
            'R_d': R_d,
            'theta_e': theta_e,
            'omega_e': omega_e,
        }


class LeeGeometricPD:
    """
    Lee et al. (2010) SE(3) geometric PD controller for baseline comparison.

    Control law:
        F_des = -K_p * e_pos - K_v * e_vel + m*g*e3 + m*x_ddot_d
        M = -K_R * theta_e - K_omega * omega_e + omega x J*omega
            - J*(omega_d^x R' R_d omega_d - R' R_d omega_d_dot)
    """

    def __init__(self, m, J, g=9.81,
                 kx=8.4, kv=7.0, kR=4.0, komega=2.0):
        self.m = m
        self.J = np.asarray(J)
        self.g = g
        self.kx = kx
        self.kv = kv
        self.kR = kR
        self.komega = komega

    def compute_control(self, state, pos_d, vel_d, yaw_d, acc_d=None,
                        omega_d=None, omega_d_dot=None):
        from models.quadrotor_se3 import QuadrotorSE3
        model = QuadrotorSE3(self.m, self.J, self.g)
        pos, vel, R, omega = model.unpack_state(state)

        if acc_d is None:
            acc_d = np.zeros(3)
        if omega_d is None:
            omega_d = np.zeros(3)
        if omega_d_dot is None:
            omega_d_dot = np.zeros(3)

        # Position errors
        e_pos = pos - pos_d
        e_vel = vel - vel_d

        # Desired force (Lee 2010, Eq. 19)
        F_des = (-self.kx * e_pos - self.kv * e_vel +
                 self.m * self.g * np.array([0., 0., 1.]) +
                 self.m * acc_d)

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)
        R_d = compute_desired_attitude(b3_des, yaw_d)

        # Attitude errors
        theta_e = compute_attitude_error(R, R_d)
        omega_e = omega - R.T @ R_d @ omega_d

        # Torque (Lee 2010, Eq. 42)
        M = (-self.kR * theta_e - self.komega * omega_e +
             np.cross(omega, self.J @ omega) -
             self.J @ (hat(omega_d) @ R.T @ R_d @ omega_d - R.T @ R_d @ omega_d_dot))

        thrust = np.dot(F_des, R @ np.array([0., 0., 1.]))

        return np.array([thrust, M[0], M[1], M[2]])
