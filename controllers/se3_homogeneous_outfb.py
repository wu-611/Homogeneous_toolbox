"""
SE(3) 几何齐次跟踪控制器 —— 输出反馈版本 (Phase 3a)
=====================================================

在全状态反馈版本的基础上，用齐次观测器 (HO) 替代速度传感器。
仅需位置测量（GPS/动作捕捉），速度由观测器估计。

架构（级联 + 输出反馈）:
    ┌──────────────┐
    │ 位置测量 y   │ (pos + noise)
    └──────┬───────┘
           │
    ┌──────┴──────────────┐
    │ 齐次观测器 (HO)      │  ← 新增：估计速度
    │ z = [ê_pos, ê_vel]  │
    └──────┬──────────────┘
           │ ê_vel
    ┌──────┴───────┐
    │ 位置 HPC     │ → u_pos [m/s²]
    └──────┬───────┘
           │
    ┌──────┴───────┐     ┌──────────┐     ┌─────────────┐
    │ 重力补偿     │ ──→ │ 期望姿态  │ ──→ │ 姿态 HPC    │
    │ F_des        │     │ 解算 R_d  │     │ (全状态)    │
    └──────────────┘     └──────────┘     └─────────────┘
                              │                   │
                              └───────────────────┘
                                      │
                              ┌───────┴────────┐
                              │  力矩/推力映射  │
                              └───────┬────────┘
                                      │
                              ┌───────┴────────┐
                              │ 四旋翼 SE(3)   │
                              └────────────────┘

关键区别 vs 全状态反馈:
    1. 速度 e_vel 被替换为观测器估计 ê_vel
    2. 观测器需位置测量 y = e_pos + noise 和已知控制 u_ctrl
    3. 使用上一时刻的 u_pos 作为观测器前馈（一阶 Euler 的显式结构）
    4. 姿态回路不变（IMU 提供完整姿态 + 角速度）

参考文献:
    - Wang Siyuan (2020) 第 4 章: 齐次观测器 + 齐次控制器实验验证
    - Polyakov (2020): lo2ho 对偶理论
    - Lee et al. (2010): SE(3) 几何控制结构
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from design.design_position_hpc import PositionHPC
from design.design_position_ho import PositionHO
from design.design_attitude_hpc import AttitudeHPC
from design.attitude_command import (compute_desired_attitude,
                                      compute_attitude_error,
                                      compute_omega_error)
from design.torque_mapping import map_virtual_to_torque


class SE3HomogeneousOutFB:
    """
    SE(3) 几何齐次跟踪控制器（输出反馈，仅需位置测量）

    使用示例:
        ctrl = SE3HomogeneousOutFB(m=1.4, J=diag(...), mu_p=-0.5, mu_a=-0.5, nu=-0.7)
        ctrl.reset(pos_measured, pos_d)  # 初始化观测器
        for each step:
            u = ctrl.compute_control(state, pos_d, vel_d, yaw_d)
    """

    def __init__(self, m, J, g=9.81, mu_p=0.0, mu_a=0.0, nu=None,
                 K_pos=None, K1_att=None, k2_att=None, max_tilt=60.0,
                 dt=0.001, noise_std=0.0):
        """
        参数:
            m, J, g: 四旋翼物理参数
            mu_p: 位置回路齐次度
            mu_a: 姿态回路齐次度
            nu:   观测器齐次度（None → 取 nu_min）
            K_pos, K1_att, k2_att: 增益参数
            max_tilt: 最大推力倾角 [deg]
            dt: 采样周期 [s]（观测器离散化步长）
            noise_std: 位置测量噪声标准差 [m]（0 = 无噪声）
        """
        self.m = m
        self.J = np.asarray(J)
        self.g = g
        self.mu_p = mu_p
        self.mu_a = mu_a
        self.dt = dt
        self.noise_std = noise_std
        self.max_tilt = np.deg2rad(max_tilt)

        # 位置回路 HPC（复用全状态版本）
        self.pos_hpc = PositionHPC(m, K_pos, mu_p)

        # 位置回路 HO（新增 — 每通道一个独立观测器）
        self.pos_ho_x = PositionHO(m, nu=nu)
        self.pos_ho_y = PositionHO(m, nu=nu)
        self.pos_ho_z = PositionHO(m, nu=nu)
        self.nu = self.pos_ho_x.nu  # 记录实际使用的 nu

        # 姿态回路 HPC（复用全状态版本）
        self.att_hpc = AttitudeHPC(J, K1_att, k2_att, mu_a)

        # 观测器状态：每个通道的 [e_pos_hat, e_vel_hat]
        self.z_x = None
        self.z_y = None
        self.z_z = None

        # 上一时刻的 HPC 输出（观测器前馈用）
        self._u_pos_prev = np.zeros(3)

    def reset(self, pos_measured=None, pos_d=None):
        """
        重置观测器状态。

        若提供位置测量和目标位置，用测量位置初始化 ê_pos，
        ê_vel 初始化为 0（冷启动速度猜测）。

        参数:
            pos_measured: 初始位置测量 (3,) [m]
            pos_d: 目标位置 (3,) [m]
        """
        if pos_measured is not None and pos_d is not None:
            e_pos = np.asarray(pos_measured).flatten() - np.asarray(pos_d).flatten()
            self.z_x = np.array([e_pos[0], 0.0])
            self.z_y = np.array([e_pos[1], 0.0])
            self.z_z = np.array([e_pos[2], 0.0])
        else:
            self.z_x = np.zeros(2)
            self.z_y = np.zeros(2)
            self.z_z = np.zeros(2)

        self._u_pos_prev = np.zeros(3)

    def compute_control(self, state, pos_d, vel_d, yaw_d, acc_d=None,
                        omega_d=None, omega_d_dot=None,
                        torque_limit=20.0, thrust_limits=(0.1, 80.0)):
        """
        计算完整的四旋翼控制输入（输出反馈版本）。

        处理流程:
            1. 测量位置（可选噪声）
            2. 观测器更新 → 估计速度误差 ê_vel
            3. 位置 HPC（用 ê_vel 替代真实 e_vel）
            4-7. 重力补偿 + 期望姿态 + 姿态 HPC + 映射（与全状态版相同）

        参数:
            state: 18维 SE(3) 状态
            pos_d, vel_d: 期望位置 [m] 和速度 [m/s]
            yaw_d: 期望偏航角 [rad]
            acc_d, omega_d, omega_d_dot: 前馈项
            torque_limit, thrust_limits: 力矩/推力限幅

        返回:
            u: [thrust, tau_x, tau_y, tau_z]
        """
        from models.quadrotor_se3 import QuadrotorSE3
        model = QuadrotorSE3(self.m, self.J, self.g)
        pos, vel, R, omega = model.unpack_state(state)

        if omega_d is None:
            omega_d = np.zeros(3)
        if omega_d_dot is None:
            omega_d_dot = np.zeros(3)

        # ====== 第1步：位置测量（可选噪声）======
        pos_measured = pos.copy()
        if self.noise_std > 0:
            pos_measured += self.noise_std * np.random.randn(3)

        # 测量位置误差（观测器的输入 y）
        e_pos_measured = pos_measured - pos_d

        # 初始化观测器（首次调用）
        if self.z_x is None:
            self.reset(pos_measured, pos_d)

        # ====== 第2步：观测器更新 → 估计速度误差 ======
        # 每个通道独立更新
        # 前馈：使用上一时刻的 HPC 输出（一阶 Euler 显式结构）
        u_fb = self._u_pos_prev

        self.z_x = self.pos_ho_x.update(self.z_x, e_pos_measured[0],
                                         u_fb[0], self.dt)
        self.z_y = self.pos_ho_y.update(self.z_y, e_pos_measured[1],
                                         u_fb[1], self.dt)
        self.z_z = self.pos_ho_z.update(self.z_z, e_pos_measured[2],
                                         u_fb[2], self.dt)

        # 提取估计的速度误差
        e_vel_hat = np.array([self.z_x[1], self.z_y[1], self.z_z[1]])

        # 使用测量位置误差 + 估计速度误差
        e_pos = e_pos_measured
        e_vel = e_vel_hat

        # ====== 第3步：位置回路 HPC ======
        u_pos = self.pos_hpc.compute_control_vector(e_pos, e_vel)
        self._u_pos_prev = u_pos.copy()

        # 叠加前馈加速度
        if acc_d is not None:
            u_pos = u_pos + np.asarray(acc_d).flatten()

        # ====== 第4步：重力补偿 + 期望推力方向 ======
        F_des = u_pos + np.array([0., 0., self.g])

        # 推力倾角限制
        F_h = np.linalg.norm(F_des[:2])
        if F_h > 1e-10:
            cos_tilt = F_des[2] / np.linalg.norm(F_des)
            if cos_tilt < np.cos(self.max_tilt) and F_des[2] > 0:
                s = F_des[2] * np.tan(self.max_tilt) / F_h
                F_des = np.array([s * F_des[0], s * F_des[1], F_des[2]])
            elif F_des[2] <= 0:
                F_des = np.array([0., 0., self.g])

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)

        # ====== 第5步：期望姿态解算 ======
        R_d = compute_desired_attitude(b3_des, yaw_d)

        # ====== 第6步：姿态误差 ======
        theta_e = compute_attitude_error(R, R_d)
        omega_e = compute_omega_error(omega, omega_d, R, R_d)

        # ====== 第7步：姿态回路（全状态，IMU提供）======
        u_hom = self.att_hpc.compute_virtual_control(theta_e, omega_e)

        # ====== 第8步：力矩映射 ======
        M = map_virtual_to_torque(u_hom, self.J, R_d, omega, omega_d, omega_d_dot)

        # ====== 第9步：推力映射 ======
        F_des_force = self.m * (u_pos + np.array([0., 0., self.g]))
        thrust = np.dot(F_des_force, R @ np.array([0., 0., 1.]))

        # ====== 第10步：饱和限制 ======
        thrust = np.clip(thrust, thrust_limits[0], thrust_limits[1])
        M = np.clip(M, -torque_limit, torque_limit)

        return np.array([thrust, M[0], M[1], M[2]])

    def get_debug_info(self, state, pos_d, vel_d, yaw_d):
        """
        返回中间信号（含观测器估计值）用于调试和日志。

        额外输出（相比全状态版本）:
            e_vel_hat:  观测器估计的速度误差
            e_vel_true: 真实速度误差（用于评估观测器精度）
            obs_error:  观测器估计误差 = e_vel_hat - e_vel_true
        """
        from models.quadrotor_se3 import QuadrotorSE3
        model = QuadrotorSE3(self.m, self.J, self.g)
        pos, vel, R, omega = model.unpack_state(state)

        e_pos = pos - pos_d
        e_vel_true = vel - vel_d

        # 估计速度误差（从观测器状态）
        e_vel_hat = np.array([
            self.z_x[1] if self.z_x is not None else 0.0,
            self.z_y[1] if self.z_y is not None else 0.0,
            self.z_z[1] if self.z_z is not None else 0.0,
        ])

        u_pos = self.pos_hpc.compute_control_vector(e_pos, e_vel_hat)
        F_des = u_pos + np.array([0., 0., self.g])
        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)
        R_d = compute_desired_attitude(b3_des, yaw_d)
        theta_e = compute_attitude_error(R, R_d)

        return {
            'e_pos': e_pos,
            'e_vel_true': e_vel_true,
            'e_vel_hat': e_vel_hat,
            'obs_error': e_vel_hat - e_vel_true,
            'u_pos': u_pos,
            'F_des': F_des,
            'b3_des': b3_des,
            'R_d': R_d,
            'theta_e': theta_e,
        }
