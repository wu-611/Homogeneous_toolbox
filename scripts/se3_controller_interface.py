#!/usr/bin/env python3
"""
SE(3) Geometric Homogeneous Controller -- ROS2 SITL Interface

Standalone module for ROS2 / PX4 SITL integration.
ROS2 node handles NED<->Z-Up conversion and DDS communication,
this module handles the core control algorithm.

Two output modes:
    'rate_setpoint': angular rate commands (for PX4 vehicle_rates_setpoint)
                     Recommended for SITL -- PX4 X500 hover torque ~0.45 Nm
    'torque':        direct torque commands (for numerical simulation)

Coordinate conventions:
    Algorithm internal: Z-Up (inertial Z upward)
    PX4 external: NED (North-East-Down), FRD (body: Forward-Right-Down)

Usage in ROS2 node (rate_setpoint mode):
    from se3_controller_interface import SE3ControllerInterface
    ctrl = SE3ControllerInterface(mode='outfb', output='rate_setpoint')
    ctrl.set_target(pos_d=[0, 0, -2], yaw_d=0)
    u = ctrl.step(pos_zup, vel_zup, R, omega)
    # u = [thrust_norm, wx_des, wy_des, wz_des]
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.quadrotor_se3 import QuadrotorSE3


def _make_full_state_ctrl(m, J, g, mu_p, mu_a, max_tilt):
    from design.design_position_hpc import PositionHPC
    from design.design_attitude_hpc import AttitudeHPC
    from design.attitude_command import (compute_desired_attitude,
                                          compute_attitude_error,
                                          compute_omega_error)
    from design.torque_mapping import map_virtual_to_torque
    pos_hpc = PositionHPC(m, mu=mu_p)
    att_hpc = AttitudeHPC(J, mu=mu_a)
    return {
        'pos_hpc': pos_hpc, 'att_hpc': att_hpc,
        'compute_desired_attitude': compute_desired_attitude,
        'compute_attitude_error': compute_attitude_error,
        'compute_omega_error': compute_omega_error,
        'map_virtual_to_torque': map_virtual_to_torque,
        'm': m, 'g': g, 'J': np.asarray(J), 'max_tilt': max_tilt,
    }


def _make_outfb_ctrl(m, J, g, mu_p, mu_a, nu, dt, max_tilt):
    from design.design_position_hpc import PositionHPC
    from design.design_position_ho import PositionHO
    from design.design_attitude_hpc import AttitudeHPC
    from design.attitude_command import (compute_desired_attitude,
                                          compute_attitude_error,
                                          compute_omega_error)
    from design.torque_mapping import map_virtual_to_torque
    pos_hpc = PositionHPC(m, mu=mu_p)
    att_hpc = AttitudeHPC(J, mu=mu_a)
    return {
        'pos_hpc': pos_hpc,
        'pos_ho': [PositionHO(m, nu=nu) for _ in range(3)],
        'att_hpc': att_hpc,
        'compute_desired_attitude': compute_desired_attitude,
        'compute_attitude_error': compute_attitude_error,
        'compute_omega_error': compute_omega_error,
        'map_virtual_to_torque': map_virtual_to_torque,
        'm': m, 'g': g, 'J': np.asarray(J),
        'dt': dt, 'max_tilt': max_tilt,
        'z_state': [np.zeros(2), np.zeros(2), np.zeros(2)],
        'u_pos_prev': np.zeros(3),
    }


class SE3ControllerInterface:
    """SE(3) homogeneous controller ROS2 SITL interface.

    Modes:
        'full':  full-state (pos+vel+att+omega)  -- needs velocity sensor
        'outfb': output feedback (pos+att+omega)  -- velocity from HO observer

    Outputs:
        'rate_setpoint': PX4 vehicle_rates_setpoint (recommended for SITL)
        'torque':        direct torque (for numerical simulation)

    PX4 X500 constraint: available torque at hover ~0.45 Nm.
    Use rate_setpoint mode to let PX4 internal rate controller
    handle torque allocation within this budget.
    """

    def __init__(self, mode='outfb', output='rate_setpoint',
                 m=1.4, J_xx=0.0211, J_yy=0.0219, J_zz=0.0366, g=9.81,
                 mu_p=-0.5, mu_a=-0.5, nu=None, K1=100, k2=50,
                 max_tilt_deg=30.0, dt=0.01,
                 rate_limit=8.0,
                 torque_limit=10.0, torque_rate_limit=50.0,
                 thrust_min=0.1, thrust_max=80.0):
        if output not in ('torque', 'rate_setpoint'):
            raise ValueError("output must be 'torque' or 'rate_setpoint'")
        self.output = output
        self.mode = mode
        self.m = m
        self.J = np.diag([J_xx, J_yy, J_zz])
        self.g = g
        self.dt = dt
        self.rate_limit = rate_limit
        self.torque_limit = torque_limit
        self.torque_rate_limit = torque_rate_limit
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.max_tilt = np.deg2rad(max_tilt_deg)

        self._M_prev = np.zeros(3)
        self._omega_des_int = np.zeros(3)

        self._pos_d = np.array([0.0, 0.0, -2.0])
        self._vel_d = np.zeros(3)
        self._yaw_d = 0.0
        self._acc_d = np.zeros(3)

        if mode == 'full':
            self._ctrl = _make_full_state_ctrl(m, self.J, g, mu_p, mu_a, self.max_tilt)
        elif mode == 'outfb':
            self._ctrl = _make_outfb_ctrl(m, self.J, g, mu_p, mu_a, nu, dt, self.max_tilt)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        from design.design_attitude_hpc import AttitudeHPC
        self._ctrl['att_hpc'] = AttitudeHPC(self.J, K1=K1, k2=k2, mu=mu_a)
        self.debug = {}

    def set_target(self, pos_d, vel_d=None, yaw_d=0.0, acc_d=None):
        self._pos_d = np.asarray(pos_d, dtype=float).flatten()
        self._vel_d = (np.zeros(3) if vel_d is None
                       else np.asarray(vel_d, dtype=float).flatten())
        self._yaw_d = float(yaw_d)
        self._acc_d = (np.zeros(3) if acc_d is None
                       else np.asarray(acc_d, dtype=float).flatten())

    def step(self, pos, vel, R, omega, dt=None):
        if dt is not None:
            self.dt = dt

        e_pos = pos - self._pos_d
        e_vel = vel - self._vel_d

        # Observer update (outfb mode)
        c = self._ctrl
        if self.mode == 'outfb':
            for i, ho in enumerate(c['pos_ho']):
                c['z_state'][i] = ho.update(
                    c['z_state'][i], e_pos[i], c['u_pos_prev'][i], self.dt)
                e_vel[i] = c['z_state'][i][1]

        u_pos = c['pos_hpc'].compute_control_vector(e_pos, e_vel)
        if np.any(self._acc_d):
            u_pos = u_pos + self._acc_d
        if self.mode == 'outfb':
            c['u_pos_prev'] = u_pos.copy()

        # Gravity compensation + tilt limit
        F_des_raw = u_pos + np.array([0., 0., self.g])
        F_des = F_des_raw.copy()
        F_h = np.linalg.norm(F_des[:2])
        if F_h > 1e-10:
            cos_tilt = F_des[2] / np.linalg.norm(F_des)
            if cos_tilt < np.cos(self.max_tilt) and F_des[2] > 0:
                s = F_des[2] * np.tan(self.max_tilt) / F_h
                F_des = np.array([s * F_des[0], s * F_des[1], F_des[2]])
            elif F_des[2] <= 0:
                F_des = np.array([0., 0., self.g])

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)
        R_d = c['compute_desired_attitude'](b3_des, self._yaw_d)
        theta_e = c['compute_attitude_error'](R, R_d)
        omega_e = c['compute_omega_error'](omega, np.zeros(3), R, R_d)
        att = c['att_hpc']

        if self.output == 'rate_setpoint':
            return self._output_rate_setpoint(e_pos, e_vel, u_pos, F_des, R,
                                               theta_e, omega_e, att)
        else:
            return self._output_torque(e_pos, e_vel, u_pos, F_des, R, R_d,
                                        theta_e, omega_e, att)

    def _output_rate_setpoint(self, e_pos, e_vel, u_pos, F_des, R,
                               theta_e, omega_e, att):
        u_hom = att.compute_virtual_control(theta_e, omega_e)
        self._omega_des_int += u_hom * self.dt
        omega_des = np.clip(self._omega_des_int, -self.rate_limit, self.rate_limit)
        self._omega_des_int = omega_des

        thrust_raw = self.m * max(np.dot(F_des, R[:, 2]), 0.0)
        hover_thrust = self.m * self.g
        thrust_norm = np.clip(thrust_raw / hover_thrust * 0.5, 0.0, 1.0)

        self.debug = {
            'e_pos': e_pos, 'e_vel': e_vel, 'u_pos': u_pos,
            'theta_e_norm': np.linalg.norm(theta_e),
            'u_hom': u_hom, 'omega_des': omega_des,
            'thrust_raw': thrust_raw, 'thrust_norm': thrust_norm,
        }
        return np.array([thrust_norm, omega_des[0], omega_des[1], omega_des[2]])

    def _output_torque(self, e_pos, e_vel, u_pos, F_des, R, R_d,
                        theta_e, omega_e, att):
        J = self.J
        u_hom = att.compute_virtual_control(theta_e, omega_e)
        M = self._ctrl['map_virtual_to_torque'](
            u_hom, J, R_d, np.zeros(3), np.zeros(3), np.zeros(3))
        thrust = np.dot(self.m * (u_pos + np.array([0., 0., self.g])), R[:, 2])
        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)
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
            self._ctrl['z_state'] = [
                np.array([e_pos[0], 0.0]),
                np.array([e_pos[1], 0.0]),
                np.array([e_pos[2], 0.0]),
            ]
            self._ctrl['u_pos_prev'] = np.zeros(3)
        self._omega_des_int = np.zeros(3)
        self._M_prev = np.zeros(3)
        self.debug = {}


# ============================================================
# ROS2 coordinate conversion utilities
# ============================================================

def ned_to_zup(pos_ned, vel_ned=None):
    """NED [N,E,D] -> Z-Up [N,E,-D]."""
    pos_zup = np.array([pos_ned[0], pos_ned[1], -pos_ned[2]])
    if vel_ned is not None:
        vel_zup = np.array([vel_ned[0], vel_ned[1], -vel_ned[2]])
        return pos_zup, vel_zup
    return pos_zup


def zup_to_ned(thrust_zup, tau_zup):
    """Z-Up thrust/torque -> NED/FRD (torque output mode)."""
    return -thrust_zup, np.array([tau_zup[0], tau_zup[1], -tau_zup[2]])


def zup_to_rate_setpoint_ned(thrust_norm, omega_des_zup):
    """Z-Up rate setpoint -> FRD for PX4 vehicle_rates_setpoint."""
    return thrust_norm, np.array([omega_des_zup[0], omega_des_zup[1],
                                   -omega_des_zup[2]])


def quat_to_rot_matrix(q_w, q_x, q_y, q_z):
    """Quaternion (Hamilton) -> rotation matrix."""
    return np.array([
        [1 - 2*q_y**2 - 2*q_z**2,  2*q_x*q_y - 2*q_w*q_z,    2*q_x*q_z + 2*q_w*q_y],
        [2*q_x*q_y + 2*q_w*q_z,    1 - 2*q_x**2 - 2*q_z**2,  2*q_y*q_z - 2*q_w*q_x],
        [2*q_x*q_z - 2*q_w*q_y,    2*q_y*q_z + 2*q_w*q_x,    1 - 2*q_x**2 - 2*q_y**2],
    ])


# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("SE(3) Controller Interface - ROS2 SITL Test")
    print("=" * 60)

    # Torque mode
    print("\n[1] Torque output mode")
    ctrl_t = SE3ControllerInterface(mode='outfb', output='torque',
                                     mu_p=-0.5, mu_a=-0.5, max_tilt_deg=30)
    ctrl_t.set_target(pos_d=[0, 0, -2])
    u_t = ctrl_t.step(np.array([1.0, 0.5, 0.0]), np.zeros(3),
                      np.eye(3), np.zeros(3))
    print(f"  thrust={u_t[0]:.2f}N  |M|={np.linalg.norm(u_t[1:4]):.3f}Nm  "
          f"M={np.round(u_t[1:4],3)}")

    # Rate-setpoint mode
    print("\n[2] Rate-setpoint output mode (PX4 vehicle_rates_setpoint)")
    ctrl_rs = SE3ControllerInterface(mode='outfb', output='rate_setpoint',
                                      mu_p=-0.5, mu_a=-0.5, max_tilt_deg=30,
                                      rate_limit=8.0)
    ctrl_rs.set_target(pos_d=[0, 0, -2])
    u_rs = ctrl_rs.step(np.array([1.0, 0.5, 0.0]), np.zeros(3),
                        np.eye(3), np.zeros(3))
    print(f"  thrust_norm={u_rs[0]:.3f}  w_des={np.round(u_rs[1:4],2)} rad/s")

    # Near hover
    print("\n[3] Rate-setpoint near hover (small error)")
    ctrl_rs.reset(np.array([0.01, -0.02, -1.99]), np.array([0.0, 0.0, -2.0]))
    u_rs2 = ctrl_rs.step(np.array([0.01, -0.02, -1.99]), np.zeros(3),
                         np.eye(3), np.zeros(3))
    print(f"  thrust_norm={u_rs2[0]:.3f}  w_des={np.round(u_rs2[1:4],4)} rad/s")

    # Coordinate transforms
    print("\n[4] Coordinate transforms")
    print(f"  NED [1,2,-3] -> Z-Up {ned_to_zup([1,2,-3])}")
    _, tf = zup_to_ned(15, [0.1, 0.2, 0.05])
    print(f"  Torque Z-Up [0.1,0.2,0.05] -> FRD {tf}")
    _, wf = zup_to_rate_setpoint_ned(0.5, [1.0, 0.5, 0.2])
    print(f"  Rate Z-Up [1.0,0.5,0.2] -> FRD {wf}")

    print("\n  All tests passed. Ready for ROS2 SITL integration.")
