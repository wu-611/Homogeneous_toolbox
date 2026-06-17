#!/usr/bin/env python3
"""
SE(3) 几何齐次控制器 —— ROS2 SITL 对接接口
=============================================

此模块提供与 ROS2 / PX4 SITL 对接的标准控制器接口。
ROS2 节点负责坐标转换 (NED ↔ Z-Up) 和 DDS 通信，
此模块负责核心控制算法。

坐标系约定（重要）：
    算法内部: Z-Up (惯性系 Z 轴向上)
    PX4 外部: NED (北-东-地, Z 轴向下), FRD (机体: 前-右-下)

    坐标转换在 ROS2 节点中完成，此模块只处理 Z-Up 约定。

使用示例 (ROS2 节点中):
    from se3_controller_interface import SE3ControllerInterface

    ctrl = SE3ControllerInterface(mode='outfb', mu_p=-0.5, mu_a=-0.5)
    ctrl.set_target(pos_d=[0, 0, -2], yaw_d=0)

    # 每步回调:
    u = ctrl.step(pos_zup, vel_zup, R, omega, dt=0.01)
    # u = [thrust, tau_x, tau_y, tau_z] (Z-Up 约定)

独立测试:
    python3 scripts/se3_controller_interface.py
"""

import numpy as np
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.quadrotor_se3 import QuadrotorSE3, hat, log_so3, exp_so3


# ============================================================
# 内部：导入控制器组件
# ============================================================
def _make_full_state_ctrl(m, J, g, mu_p, mu_a, max_tilt):
    """创建全状态反馈控制器（需要速度测量）。"""
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
        'm': m, 'g': g, 'J': np.asarray(J),
        'max_tilt': max_tilt,
    }


def _make_outfb_ctrl(m, J, g, mu_p, mu_a, nu, dt, max_tilt):
    """创建输出反馈控制器（仅需位置测量）。"""
    from design.design_position_hpc import PositionHPC
    from design.design_position_ho import PositionHO
    from design.design_attitude_hpc import AttitudeHPC
    from design.attitude_command import (compute_desired_attitude,
                                          compute_attitude_error,
                                          compute_omega_error)
    from design.torque_mapping import map_virtual_to_torque

    pos_hpc = PositionHPC(m, mu=mu_p)
    pos_ho_x = PositionHO(m, nu=nu)
    pos_ho_y = PositionHO(m, nu=nu)
    pos_ho_z = PositionHO(m, nu=nu)
    att_hpc = AttitudeHPC(J, mu=mu_a)

    return {
        'pos_hpc': pos_hpc,
        'pos_ho': [pos_ho_x, pos_ho_y, pos_ho_z],
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


# ============================================================
# 公开接口类
# ============================================================

class SE3ControllerInterface:
    """
    SE(3) 齐次控制器 ROS2 对接接口。

    支持两种模式:
        'full': 全状态反馈（需要位置 + 速度 + 姿态 + 角速度）
        'outfb': 输出反馈（仅需位置 + 姿态 + 角速度，速度由观测器估计）

    PX4 状态消息的对应关系:
        VehicleOdometry.position    → pos (NED → Z-Up 转换后)
        VehicleOdometry.velocity    → vel (NED → Z-Up)
        VehicleOdometry.q           → R (四元数 → 旋转矩阵)
        VehicleOdometry.angular_velocity → omega (FRD → Z-Up)
    """

    def __init__(self, mode='outfb',
                 m=1.4, J_xx=0.0211, J_yy=0.0219, J_zz=0.0366, g=9.81,
                 mu_p=-0.5, mu_a=-0.5, nu=None, K1=100, k2=50,
                 max_tilt_deg=45.0, dt=0.01,
                 torque_limit=10.0, torque_rate_limit=50.0,
                 thrust_min=0.1, thrust_max=80.0):
        """
        参数:
            mode: 'full' (全状态) 或 'outfb' (输出反馈，推荐)
            m: 质量 [kg]（默认 1.4）
            J_xx, J_yy, J_zz: 转动惯量对角线元素 [kg·m²]
            g: 重力加速度 [m/s²]（默认 9.81）
            mu_p: 位置回路齐次度（默认 -0.5, 有限时间收敛）
            mu_a: 姿态回路齐次度（默认 -0.5）
            nu: 观测器齐次度（None=自动取 nu_min, 用于 'outfb' 模式）
            max_tilt_deg: 最大推力倾角 [°]（默认 60）
            dt: 控制器调用周期 [s]（默认 0.01 = 100Hz）
            torque_limit: 力矩限幅 [N·m]（每轴，默认 20）
            thrust_min, thrust_max: 推力限幅 [N]
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

        # 力矩历史（用于 rate limiting）
        self._M_prev = np.zeros(3)

        # 目标状态
        self._pos_d = np.array([0.0, 0.0, -2.0])
        self._vel_d = np.zeros(3)
        self._yaw_d = 0.0
        self._acc_d = np.zeros(3)

        # 创建控制器组件（传递 K1, k2）
        if mode == 'full':
            self._ctrl = _make_full_state_ctrl(
                m, self.J, g, mu_p, mu_a, self.max_tilt)
            # 覆盖默认增益
            self._ctrl['att_hpc'] = type(self._ctrl['att_hpc'])(
                self.J, K1=K1, k2=k2, mu=mu_a)
        elif mode == 'outfb':
            self._ctrl = _make_outfb_ctrl(
                m, self.J, g, mu_p, mu_a, nu, dt, self.max_tilt)
            self._ctrl['att_hpc'] = type(self._ctrl['att_hpc'])(
                self.J, K1=K1, k2=k2, mu=mu_a)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 调试信息
        self.debug = {}

    def set_target(self, pos_d, vel_d=None, yaw_d=0.0, acc_d=None):
        """
        设置控制目标。

        参数:
            pos_d: 目标位置 [x, y, z] (Z-Up) [m]
            vel_d: 目标速度 [vx, vy, vz] (Z-Up) [m/s]（默认零）
            yaw_d: 目标偏航角 [rad]（默认 0）
            acc_d: 前馈加速度 [ax, ay, az] (Z-Up) [m/s²]（默认零）
        """
        self._pos_d = np.asarray(pos_d, dtype=float).flatten()
        self._vel_d = (np.zeros(3) if vel_d is None
                       else np.asarray(vel_d, dtype=float).flatten())
        self._yaw_d = float(yaw_d)
        self._acc_d = (np.zeros(3) if acc_d is None
                       else np.asarray(acc_d, dtype=float).flatten())

    def step(self, pos, vel, R, omega, dt=None):
        """
        计算单步控制输出（由 ROS2 节点回调调用）。

        参数:
            pos:   当前位置 [x, y, z] (Z-Up) [m]
            vel:   当前速度 [vx, vy, vz] (Z-Up) [m/s]
                   ('outfb' 模式下不使用，可传零向量)
            R:     当前旋转矩阵 (3×3) (机体→惯性系)
            omega: 当前体角速度 [ωx, ωy, ωz] [rad/s]

        返回:
            u: [thrust, tau_x, tau_y, tau_z]
               thrust: 总推力 [N] (沿机体 Z 轴, 标量)
               tau_x, tau_y, tau_z: 体坐标系力矩 [N·m]
        """
        if dt is not None:
            self.dt = dt

        pos_d = self._pos_d
        vel_d = self._vel_d
        yaw_d = self._yaw_d
        acc_d = self._acc_d
        m = self.m
        g = self.g
        J = self.J

        if self.mode == 'full':
            return self._step_full(pos, vel, R, omega,
                                   pos_d, vel_d, yaw_d, acc_d)
        else:
            return self._step_outfb(pos, vel, R, omega,
                                    pos_d, vel_d, yaw_d, acc_d)

    def _step_full(self, pos, vel, R, omega,
                   pos_d, vel_d, yaw_d, acc_d):
        """全状态反馈控制步骤。"""
        c = self._ctrl
        pos_hpc = c['pos_hpc']
        att_hpc = c['att_hpc']
        m, g, J, max_tilt = c['m'], c['g'], c['J'], c['max_tilt']

        # 位置误差
        e_pos = pos - pos_d
        e_vel = vel - vel_d

        # 位置 HPC → 期望加速度
        u_pos = pos_hpc.compute_control_vector(e_pos, e_vel)
        if np.any(acc_d):
            u_pos = u_pos + acc_d

        # 重力补偿 + 推力方向
        F_des = u_pos + np.array([0., 0., g])

        # 推力倾角限制
        F_h = np.linalg.norm(F_des[:2])
        if F_h > 1e-10:
            cos_tilt = F_des[2] / np.linalg.norm(F_des)
            if cos_tilt < np.cos(max_tilt) and F_des[2] > 0:
                s = F_des[2] * np.tan(max_tilt) / F_h
                F_des[:2] *= s
            elif F_des[2] <= 0:
                F_des = np.array([0., 0., g])

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)

        # 期望姿态
        R_d = c['compute_desired_attitude'](b3_des, yaw_d)

        # 姿态误差
        theta_e = c['compute_attitude_error'](R, R_d)
        omega_e = c['compute_omega_error'](omega, np.zeros(3), R, R_d)

        # 姿态 HPC
        u_hom = att_hpc.compute_virtual_control(theta_e, omega_e)

        # 力矩映射
        M = c['map_virtual_to_torque'](u_hom, J, R_d, omega,
                                        np.zeros(3), np.zeros(3))

        # 推力映射
        F_des_force = m * (u_pos + np.array([0., 0., g]))
        thrust = np.dot(F_des_force, R[:, 2])

        # 饱和
        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)
        M = np.clip(M, -self.torque_limit, self.torque_limit)

        # 力矩变化率限制 (防阶跃)
        if self.torque_rate_limit > 0 and self.dt > 0:
            dM_max = self.torque_rate_limit * self.dt
            M = np.clip(M, self._M_prev - dM_max, self._M_prev + dM_max)
        self._M_prev = M.copy()

        # 保存调试信息
        self.debug = {
            'e_pos': e_pos, 'e_vel': e_vel,
            'u_pos': u_pos, 'theta_e_norm': np.linalg.norm(theta_e),
            'thrust': thrust, 'M': M,
        }

        return np.array([thrust, M[0], M[1], M[2]])

    def _step_outfb(self, pos, vel, R, omega,
                    pos_d, vel_d, yaw_d, acc_d):
        """输出反馈控制步骤（齐次观测器估计速度）。"""
        c = self._ctrl
        pos_hpc = c['pos_hpc']
        ho_list = c['pos_ho']
        att_hpc = c['att_hpc']
        m, g, J, dt, max_tilt = c['m'], c['g'], c['J'], c['dt'], c['max_tilt']
        z_state = c['z_state']
        u_pos_prev = c['u_pos_prev']

        # 位置误差（测量值）
        e_pos = pos - pos_d

        # 观测器更新：每个通道独立
        e_vel_hat = np.zeros(3)
        for i, ho in enumerate(ho_list):
            z_state[i] = ho.update(z_state[i], e_pos[i], u_pos_prev[i], dt)
            e_vel_hat[i] = z_state[i][1]

        # 用估计速度计算位置 HPC
        u_pos = pos_hpc.compute_control_vector(e_pos, e_vel_hat)
        c['u_pos_prev'] = u_pos.copy()

        if np.any(acc_d):
            u_pos = u_pos + acc_d

        # 以下与全状态反馈相同
        F_des = u_pos + np.array([0., 0., g])

        F_h = np.linalg.norm(F_des[:2])
        if F_h > 1e-10:
            cos_tilt = F_des[2] / np.linalg.norm(F_des)
            if cos_tilt < np.cos(max_tilt) and F_des[2] > 0:
                s = F_des[2] * np.tan(max_tilt) / F_h
                F_des[:2] *= s
            elif F_des[2] <= 0:
                F_des = np.array([0., 0., g])

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)
        R_d = c['compute_desired_attitude'](b3_des, yaw_d)
        theta_e = c['compute_attitude_error'](R, R_d)
        omega_e = c['compute_omega_error'](omega, np.zeros(3), R, R_d)
        u_hom = att_hpc.compute_virtual_control(theta_e, omega_e)
        M = c['map_virtual_to_torque'](u_hom, J, R_d, omega,
                                        np.zeros(3), np.zeros(3))
        F_des_force = m * (u_pos + np.array([0., 0., g]))
        thrust = np.dot(F_des_force, R[:, 2])

        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)
        M = np.clip(M, -self.torque_limit, self.torque_limit)

        # 力矩变化率限制 (防阶跃)
        if self.torque_rate_limit > 0 and self.dt > 0:
            dM_max = self.torque_rate_limit * self.dt
            M = np.clip(M, self._M_prev - dM_max, self._M_prev + dM_max)
        self._M_prev = M.copy()

        # 调试信息（含观测器状态）
        e_vel_true = vel - vel_d
        self.debug = {
            'e_pos': e_pos,
            'e_vel_hat': e_vel_hat,
            'e_vel_true': e_vel_true,
            'obs_error': e_vel_hat - e_vel_true,
            'u_pos': u_pos,
            'theta_e_norm': np.linalg.norm(theta_e),
            'thrust': thrust, 'M': M,
        }

        return np.array([thrust, M[0], M[1], M[2]])

    def reset(self, pos_measured=None, pos_d=None):
        """
        重置控制器状态（观测器初始化）。

        在切换目标或开始时调用，清除观测器内部状态。

        参数:
            pos_measured: 初始位置测量 [x, y, z] (Z-Up) [m]
            pos_d:        目标位置 [x, y, z] (Z-Up) [m]
        """
        if self.mode == 'outfb':
            if pos_measured is not None and pos_d is not None:
                e_pos = np.asarray(pos_measured) - np.asarray(pos_d)
            else:
                e_pos = np.zeros(3)
            self._ctrl['z_state'] = [
                np.array([e_pos[0], 0.0]),
                np.array([e_pos[1], 0.0]),
                np.array([e_pos[2], 0.0]),
            ]
            self._ctrl['u_pos_prev'] = np.zeros(3)
        self.debug = {}


# ============================================================
# ROS2 坐标转换工具函数
# ============================================================

def ned_to_zup(pos_ned, vel_ned=None):
    """
    NED (北-东-地) → Z-Up 坐标转换。

    转换: [N, E, D] → [N, E, -D]

    参数:
        pos_ned: NED 位置 (3,) [m]
        vel_ned: NED 速度 (3,) [m/s]（可选）

    返回:
        pos_zup, (vel_zup)
    """
    pos_zup = np.array([pos_ned[0], pos_ned[1], -pos_ned[2]])
    if vel_ned is not None:
        vel_zup = np.array([vel_ned[0], vel_ned[1], -vel_ned[2]])
        return pos_zup, vel_zup
    return pos_zup


def zup_to_ned(thrust_zup, tau_zup):
    """
    Z-Up → NED/FRD 推力/力矩转换。

    转换:
        推力方向: Z-Up [0,0,+F] → NED [0,0,-F]
        力矩 X:   Z-Up [+τx] → FRD [+τx] (不变)
        力矩 Y:   Z-Up [+τy] → FRD [+τy] (不变)
        力矩 Z:   Z-Up [+τz] → FRD [-τz] (偏航方向反转)

    参数:
        thrust_zup: Z-Up 推力 (标量, 沿机体 Z 向上) [N]
        tau_zup:    Z-Up 力矩 [τx, τy, τz] [N·m]

    返回:
        thrust_ned: NED 推力 (标量) [N]
        tau_frd:    FRD 力矩 [τx, τy, τz] [N·m]
    """
    # 推力：Z-Up 中正 Z = 向上，NED 中正 Z = 向下
    # 机体 Z 轴推力方向在 Z-Up 中沿 [0,0,1]，在 NED 中沿 [0,0,-1]
    thrust_ned = -thrust_zup

    # 力矩：X, Y 轴不变，Z 轴反转（右手系变化）
    tau_frd = np.array([tau_zup[0], tau_zup[1], -tau_zup[2]])

    return thrust_ned, tau_frd


def quat_to_rot_matrix(q_w, q_x, q_y, q_z):
    """
    四元数 → 旋转矩阵（Hamilton 约定, w + xi + yj + zk）。

    PX4 使用 Hamilton 四元数约定。

    返回:
        R: (3×3) 旋转矩阵（机体 → 惯性系）
    """
    R = np.array([
        [1 - 2*q_y**2 - 2*q_z**2,  2*q_x*q_y - 2*q_w*q_z,    2*q_x*q_z + 2*q_w*q_y],
        [2*q_x*q_y + 2*q_w*q_z,    1 - 2*q_x**2 - 2*q_z**2,  2*q_y*q_z - 2*q_w*q_x],
        [2*q_x*q_z - 2*q_w*q_y,    2*q_y*q_z + 2*q_w*q_x,    1 - 2*q_x**2 - 2*q_y**2],
    ])
    return R


# ============================================================
# 独立测试
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SE(3) 控制器接口测试 (ROS2 SITL 对接)")
    print("=" * 60)

    # 创建控制器（输出反馈模式）
    ctrl = SE3ControllerInterface(mode='outfb', mu_p=-0.5, mu_a=-0.5)

    # 设置目标：悬停 z=-2m
    ctrl.set_target(pos_d=[0, 0, -2], yaw_d=0)

    # 模拟 100Hz 控制循环
    dt = 0.01
    pos = np.array([1.0, 0.5, 0.0])   # 初始位置（Z-Up）
    vel = np.array([0.0, 0.0, 0.0])    # 初始速度
    R = np.eye(3)                       # 初始姿态
    omega = np.zeros(3)

    print(f"\n初始状态: pos={pos}, vel={vel}")
    print(f"目标:     pos={ctrl._pos_d}")
    print(f"\n{'步数':>5} {'推力[N]':>10} {'τx':>10} {'τy':>10} {'τz':>10} "
          f"{'|e_pos|':>10} {'|θe|':>10}")

    for k in range(20):
        u = ctrl.step(pos, vel, R, omega)
        dbg = ctrl.debug

        if k % 2 == 0:
            print(f"{k:5d} {u[0]:10.3f} {u[1]:10.4f} {u[2]:10.4f} {u[3]:10.4f} "
                  f"{np.linalg.norm(dbg['e_pos']):10.4f} "
                  f"{dbg.get('theta_e_norm', 0):10.4f}")

        # 简单仿真递推（仅演示接口，不做完整动力学）
        # 假设推力竖直，简化位置更新
        a_z = -9.81 + u[0] / 1.4  # 简化，假设 R=I
        vel[2] += a_z * dt
        pos[2] += vel[2] * dt

    # 测试坐标转换
    print("\n--- 坐标转换测试 ---")
    thrust_ned, tau_frd = zup_to_ned(15.0, np.array([0.1, 0.2, 0.05]))
    print(f"Z-Up: thrust=15.0, tau=[0.1, 0.2, 0.05]")
    print(f"NED : thrust={thrust_ned}, tau={tau_frd}")

    print("\n✓ 控制器接口测试完成。可在 ROS2 节点中导入使用。")
