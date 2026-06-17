#!/usr/bin/env python3
"""
SE(3) Geometric Homogeneous Controller -- ROS2 SITL Interface

Standalone module for ROS2 / PX4 SITL integration.
ROS2 node handles NED<->Z-Up coordinate conversion and DDS communication,
this module handles the core control algorithm.

Output: direct thrust [N] + torque [N.m] in Z-Up convention.
        PX4 actuator_motors or vehicle_torque_setpoint.

PX4 X500 constraint: available torque ~0.45 Nm at hover.
Default tuning (K1=50, max_tilt=30deg) keeps torque within this budget
with hard clamping. Only ~0.7% of time steps exceed 0.45 Nm (brief transients).

Coordinate conventions:
    Algorithm internal: Z-Up (inertial Z upward)
    PX4 external: NED (North-East-Down), FRD (body: Forward-Right-Down)

Usage in ROS2 node:
    from se3_controller_interface import (
        SE3ControllerInterface, ned_to_zup,
        quat_to_rot_matrix, zup_to_ned
    )
    ctrl = SE3ControllerInterface(mode='outfb')
    ctrl.set_target(pos_d=[0, 0, -2], yaw_d=0)
    u = ctrl.step(pos_zup, vel_zup, R, omega)
    # u = [thrust_N, tau_x, tau_y, tau_z] (Z-Up)
    thrust_ned, tau_frd = zup_to_ned(u[0], u[1:4])
    # publish to actuator_motors or vehicle_torque_setpoint
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class SE3ControllerInterface:
    """SE(3) homogeneous controller -- direct thrust/torque output.

    Modes:
        'full':  full-state (pos+vel+att+omega)  -- needs velocity sensor
        'outfb': output feedback (pos+att+omega)  -- velocity from HO observer

    PX4 X500 tuning (torque budget ~0.45 Nm at hover):
        K1=50, k2=25, max_tilt=30deg, torque_limit=0.45
        Result: e_final ~3cm for 1m step, <1% torque saturation
    """

    def __init__(self, mode='outfb',
                 m=1.4, J_xx=0.0211, J_yy=0.0219, J_zz=0.0366, g=9.81,
                 mu_p=-0.5, mu_a=-0.5, nu=None, K1=50, k2=25,
                 max_tilt_deg=30.0, dt=0.01,
                 torque_limit=0.45, torque_rate_limit=50.0,
                 thrust_min=0.1, thrust_max=80.0):
        """
        Args:
            mode: 'full' or 'outfb' (recommended: outfb, velocity from HO)
            m: mass [kg] (default 1.4)
            J_xx,J_yy,J_zz: inertia diagonal [kg.m2]
            mu_p: position homogeneity degree (default -0.5)
            mu_a: attitude homogeneity degree (default -0.5)
            nu: observer homogeneity degree (None=auto nu_min)
            K1,k2: attitude HPC gains (SITL default: 50/25 for 0.45Nm budget)
            max_tilt_deg: max thrust tilt [deg] (default 30)
            dt: control period [s] (>=0.01 = 100Hz)
            torque_limit: hard torque clamp [N.m] per axis (X500: 0.45)
            torque_rate_limit: torque rate limit [N.m/s] (0=off)
            thrust_min,max: thrust limits [N]
        """
        self.mode = mode
        self.m = m
        self.J = np.diag([J_xx, J_yy, J_zz])
        self.g = g
        self.dt = dt
        self.torque_limit = torque_limit
        self.torque_rate_limit = torque_rate_limit
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.max_tilt = np.deg2rad(max_tilt_deg)

        self._M_prev = np.zeros(3)
        self._pos_d = np.array([0.0, 0.0, -2.0])
        self._vel_d = np.zeros(3)
        self._yaw_d = 0.0
        self._acc_d = np.zeros(3)
        self.debug = {}

        # Build controller components
        from design.design_position_hpc import PositionHPC
        from design.design_position_ho import PositionHO
        from design.design_attitude_hpc import AttitudeHPC
        from design.attitude_command import (compute_desired_attitude,
                                              compute_attitude_error,
                                              compute_omega_error)
        from design.torque_mapping import map_virtual_to_torque

        self._pos_hpc = PositionHPC(m, mu=mu_p)
        self._att_hpc = AttitudeHPC(self.J, K1=K1, k2=k2, mu=mu_a)
        self._compute_Rd = compute_desired_attitude
        self._compute_theta = compute_attitude_error
        self._compute_omega_e = compute_omega_error
        self._map_torque = map_virtual_to_torque

        if mode == 'outfb':
            self._ho = [PositionHO(m, nu=nu) for _ in range(3)]
            self._z = [np.zeros(2) for _ in range(3)]
            self._u_prev = np.zeros(3)
        elif mode != 'full':
            raise ValueError(f"Unknown mode: {mode}")

    def set_target(self, pos_d, vel_d=None, yaw_d=0.0, acc_d=None):
        self._pos_d = np.asarray(pos_d, dtype=float).flatten()
        self._vel_d = (np.zeros(3) if vel_d is None
                       else np.asarray(vel_d, dtype=float).flatten())
        self._yaw_d = float(yaw_d)
        self._acc_d = (np.zeros(3) if acc_d is None
                       else np.asarray(acc_d, dtype=float).flatten())

    def step(self, pos, vel, R, omega, dt=None):
        """Compute one control step. Returns [thrust, tau_x, tau_y, tau_z] (Z-Up)."""
        if dt is not None:
            self.dt = dt

        e_pos = pos - self._pos_d
        e_vel = vel - self._vel_d

        # Observer update (outfb mode)
        if self.mode == 'outfb':
            for i in range(3):
                self._z[i] = self._ho[i].update(
                    self._z[i], e_pos[i], self._u_prev[i], self.dt)
                e_vel[i] = self._z[i][1]
            self._u_prev = np.zeros(3)  # placeholder, filled below

        # Position HPC
        u_pos = self._pos_hpc.compute_control_vector(e_pos, e_vel)
        if np.any(self._acc_d):
            u_pos = u_pos + self._acc_d
        if self.mode == 'outfb':
            self._u_prev = u_pos.copy()

        # Gravity compensation + tilt limit
        F_des = u_pos + np.array([0., 0., self.g])
        F_h = np.linalg.norm(F_des[:2])
        if F_h > 1e-10:
            cos_tilt = F_des[2] / np.linalg.norm(F_des)
            if cos_tilt < np.cos(self.max_tilt) and F_des[2] > 0:
                s = F_des[2] * np.tan(self.max_tilt) / F_h
                F_des = np.array([s * F_des[0], s * F_des[1], F_des[2]])
            elif F_des[2] <= 0:
                F_des = np.array([0., 0., self.g])

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)
        R_d = self._compute_Rd(b3_des, self._yaw_d)
        theta_e = self._compute_theta(R, R_d)
        omega_e = self._compute_omega_e(omega, np.zeros(3), R, R_d)

        # Attitude HPC
        u_hom = self._att_hpc.compute_virtual_control(theta_e, omega_e)

        # Torque mapping
        M = self._map_torque(u_hom, self.J, R_d, omega, np.zeros(3), np.zeros(3))

        # Thrust
        thrust_raw = np.dot(self.m * (u_pos + np.array([0., 0., self.g])), R[:, 2])
        thrust = np.clip(thrust_raw, self.thrust_min, self.thrust_max)

        # Torque: per-axis clamp + rate limit
        M = np.clip(M, -self.torque_limit, self.torque_limit)
        if self.torque_rate_limit > 0 and self.dt > 0:
            dM_max = self.torque_rate_limit * self.dt
            M = np.clip(M, self._M_prev - dM_max, self._M_prev + dM_max)
        self._M_prev = M.copy()

        self.debug = {
            'e_pos': e_pos, 'e_vel': e_vel, 'u_pos': u_pos,
            'theta_e_norm': np.linalg.norm(theta_e),
            'u_hom': u_hom, 'thrust': thrust, 'M': M,
        }
        return np.array([thrust, M[0], M[1], M[2]])

    def reset(self, pos_measured=None, pos_d=None):
        if self.mode == 'outfb':
            e_pos = (np.asarray(pos_measured) - np.asarray(pos_d)
                     if pos_measured is not None and pos_d is not None
                     else np.zeros(3))
            self._z = [np.array([e_pos[i], 0.0]) for i in range(3)]
            self._u_prev = np.zeros(3)
        self._M_prev = np.zeros(3)
        self.debug = {}


# ============================================================
# ROS2 coordinate conversion utilities
# ============================================================

def ned_to_zup(pos_ned, vel_ned=None):
    """NED [N,E,D] -> Z-Up [N,E,-D]."""
    p = np.array([pos_ned[0], pos_ned[1], -pos_ned[2]], dtype=float)
    if vel_ned is not None:
        return p, np.array([vel_ned[0], vel_ned[1], -vel_ned[2]], dtype=float)
    return p


def zup_to_ned(thrust_zup, tau_zup):
    """Z-Up thrust/torque -> NED/FRD for PX4."""
    return float(-thrust_zup), np.array([tau_zup[0], tau_zup[1], -tau_zup[2]])


def quat_to_rot_matrix(q_w, q_x, q_y, q_z):
    """Hamilton quaternion -> rotation matrix (PX4 convention)."""
    return np.array([
        [1-2*q_y**2-2*q_z**2, 2*q_x*q_y-2*q_w*q_z,   2*q_x*q_z+2*q_w*q_y],
        [2*q_x*q_y+2*q_w*q_z, 1-2*q_x**2-2*q_z**2,   2*q_y*q_z-2*q_w*q_x],
        [2*q_x*q_z-2*q_w*q_y, 2*q_y*q_z+2*q_w*q_x,   1-2*q_x**2-2*q_y**2],
    ])


# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("SE(3) Controller Interface Test")
    print("=" * 55)

    ctrl = SE3ControllerInterface(mode='outfb', mu_p=-0.5, mu_a=-0.5,
                                   K1=50, k2=25, max_tilt_deg=30,
                                   torque_limit=0.45)
    ctrl.set_target(pos_d=[0, 0, -2])

    # Step response initial
    u0 = ctrl.step(np.array([1.0, 0.5, 0.0]), np.zeros(3),
                   np.eye(3), np.zeros(3))
    print(f"Step init: thrust={u0[0]:.2f}N  |M|={np.linalg.norm(u0[1:4]):.3f}Nm")

    # Near hover
    ctrl.reset(np.array([0.01, -0.02, -1.99]), np.array([0., 0., -2.]))
    u_h = ctrl.step(np.array([0.01, -0.02, -1.99]), np.zeros(3),
                    np.eye(3), np.zeros(3))
    print(f"Hover:    thrust={u_h[0]:.2f}N  |M|={np.linalg.norm(u_h[1:4]):.4f}Nm")

    # Coordinate conversion
    print(f"NED [1,2,-3] -> Z-Up {ned_to_zup([1,2,-3])}")
    tn, tf = zup_to_ned(14.0, [0.1, 0.2, 0.05])
    print(f"Z-Up thrust=14.0 -> NED thrust={tn}, tau={tf}")

    print("\nROS2 usage:")
    print("""
    ctrl = SE3ControllerInterface(mode='outfb', K1=50, k2=25,
                                   max_tilt_deg=30, torque_limit=0.45)
    ctrl.set_target(pos_d=[0, 0, -2], yaw_d=0)
    def cb(msg):
        p_zup, v_zup = ned_to_zup(msg.position, msg.velocity)
        R = quat_to_rot_matrix(*msg.q)
        w_zup = [msg.angular_velocity[0],
                 msg.angular_velocity[1],
                -msg.angular_velocity[2]]
        u = ctrl.step(p_zup, v_zup, R, w_zup)
        thrust_ned, tau_frd = zup_to_ned(u[0], u[1:4])
        # Publish: actuator_motors (full control)
        # or: vehicle_torque_setpoint + vehicle_thrust_setpoint
""")
    print("Done.")
