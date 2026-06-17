"""
SE(3) 几何齐次跟踪控制器 —— 全状态反馈版本
=============================================

将 Lee et al. (2010) 的 SE(3) 几何 PD 控制器中的线性项
替换为齐次控制器 (HPC)，实现：
    - mu_p = 0, mu_a = 0 : 指数收敛（Phase 1）
    - mu_p < 0, mu_a < 0 : 有限时间收敛（Phase 2）

架构（级联控制）:
    ┌─────────────┐     ┌──────────┐     ┌─────────────┐
    │ 位置 HPC    │ ──→ │ 期望姿态  │ ──→ │ 姿态 HPC    │
    │ (加速度指令) │     │ 解算 R_d  │     │ (角加速度)   │
    └─────────────┘     └──────────┘     └─────────────┘
          │                                     │
          │  u_pos [m/s²]                       │  u_hom [rad/s²]
          │                                     │
          └───────┬─────────────────────────────┘
                  │
          ┌───────┴────────┐
          │  力矩/推力映射  │
          │  M, thrust     │
          └───────┬────────┘
                  │
          ┌───────┴────────┐
          │ 四旋翼 SE(3)   │
          │ 动力学 (RK4)   │
          └────────────────┘

参考文献:
    - Lee, Leok & McClamroch (2010): SE(3) 几何跟踪控制
    - Zhou, Polyakov & Zheng (2023): SO(3) 齐次姿态控制
    - Polyakov (2020): 广义齐次系统理论

重要实现细节:
    1. 位置 HPC 输出是"加速度" [m/s²]，不是"力" [N]
       → 推力计算: F = m * (u_pos + g*e3)，然后投影到机体 Z 轴
    2. 姿态 HPC 输出是"角加速度" [rad/s²]，不是"力矩" [N·m]
       → 力矩计算: M = J * u_hom（近似，加陀螺补偿）
    3. 齐次控制器使用齐次范数 ||x||_d 进行非线性缩放
       → μ=0 时退化为线性反馈（均匀膨胀）
       → μ<0 时实现有限时间收敛（非均匀膨胀）
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
    SE(3) 几何齐次跟踪控制器（全状态反馈）

    替换 Lee et al. (2010) 几何控制器中的线性 PD 项为齐次控制器，
    同时作用于位置回路和姿态回路。

    使用示例:
        ctrl = SE3HomogeneousController(m=1.4, J=np.diag([0.0211, 0.0219, 0.0366]),
                                         mu_p=-0.5, mu_a=-0.5)
        u = ctrl.compute_control(state, pos_d, vel_d, yaw_d)
        # u = [thrust(N), tau_x(N·m), tau_y(N·m), tau_z(N·m)]
    """

    def __init__(self, m, J, g=9.81, mu_p=0.0, mu_a=0.0,
                 K_pos=None, K1_att=None, k2_att=None, max_tilt=60.0):
        """
        参数:
            m: 飞行器质量 [kg]
            J: 转动惯量矩阵 (3×3) [kg·m²]
            g: 重力加速度 [m/s²]
            mu_p: 位置回路齐次度（0: 指数, <0: 有限时间）
            mu_a: 姿态回路齐次度
            K_pos: 位置线性增益（默认自动配置）
            K1_att: 姿态比例增益（默认 200·I₃）
            k2_att: 姿态阻尼增益（默认 100）
            max_tilt: 最大允许推力倾角 [deg]（默认 60°）
        """
        self.m = m
        self.J = np.asarray(J)
        self.g = g
        self.mu_p = mu_p
        self.mu_a = mu_a
        self.max_tilt = np.deg2rad(max_tilt)

        # 设计位置回路 HPC（三个独立的双积分器通道）
        self.pos_hpc = PositionHPC(m, K_pos, mu_p)

        # 设计姿态回路 HPC（so(3) 指数坐标，6维状态）
        self.att_hpc = AttitudeHPC(J, K1_att, k2_att, mu_a)

    def compute_control(self, state, pos_d, vel_d, yaw_d, acc_d=None,
                        omega_d=None, omega_d_dot=None,
                        torque_limit=20.0, thrust_limits=(0.1, 80.0)):
        """
        计算完整的四旋翼控制输入。

        处理流程:
            1. 从 18 维状态向量解包位置、速度、姿态、角速度
            2. 位置回路: e_pos, e_vel → HPC → 期望加速度 u_pos [m/s²]
            3. 重力补偿: F_des = u_pos + g*e3（比力，单位 N/kg）
            4. 期望姿态: F_des 方向 → b3_des → R_d
            5. 姿态回路: θ_e, ω_e → HPC → 虚拟控制 u_hom [rad/s²]
            6. 力矩映射: u_hom → M [N·m]
            7. 推力映射: F_des 投影到机体 Z 轴 → thrust [N]
            8. 饱和限制: 推力和力矩限幅

        参数:
            state: 18维 SE(3) 状态 [pos(3), vel(3), R(9), omega(3)]
            pos_d, vel_d: 期望位置 [m] 和速度 [m/s]
            yaw_d: 期望偏航角 [rad]
            acc_d: 前馈加速度 [m/s²]（如向心加速度），叠加到位置HPC输出
            omega_d: 期望体角速度（默认零）
            omega_d_dot: 期望体角加速度（默认零）
            torque_limit: 力矩限幅 [N·m]（每轴）
            thrust_limits: (最小推力, 最大推力) [N]

        返回:
            u: [thrust, tau_x, tau_y, tau_z]
        """
        from models.quadrotor_se3 import QuadrotorSE3
        model = QuadrotorSE3(self.m, self.J, self.g)
        # 解包 18 维状态向量
        pos, vel, R, omega = model.unpack_state(state)

        if omega_d is None:
            omega_d = np.zeros(3)
        if omega_d_dot is None:
            omega_d_dot = np.zeros(3)

        # ====== 第1步：位置回路 ======
        # 计算位置和速度误差
        e_pos = pos - pos_d
        e_vel = vel - vel_d
        # 位置 HPC 输出：期望加速度（惯性系，Z-up）
        u_pos = self.pos_hpc.compute_control_vector(e_pos, e_vel)
        # 叠加前馈加速度（如向心加速度），减少反馈负担
        if acc_d is not None:
            u_pos = u_pos + np.asarray(acc_d).flatten()

        # ====== 第2步：重力补偿 + 期望推力方向 ======
        # F_des 是比力 [N/kg] = [m/s²]，即单位质量的期望力
        F_des = u_pos + np.array([0., 0., self.g])

        # ---- 推力倾角限制（参考 Elastic-Tracker, Lee 2010）----
        # 目的: 防止大误差时 b3_des 过度偏离竖直方向
        #   → 机体过度倾斜 → 垂直推力不足 → Z轴下坠 → 正反馈发散
        # 原理: 保持力的方向，但缩放水平分量使倾角 ≤ max_tilt
        # 公式: F_h_new = s·F_h,  s.t. F_z / |F_new| = cos(max_tilt)
        #       s = F_z · tan(max_tilt) / |F_h|
        max_tilt = self.max_tilt  # 最大允许倾角 [rad]
        F_h = np.linalg.norm(F_des[:2])
        if F_h > 1e-10:
            cos_tilt = F_des[2] / np.linalg.norm(F_des)
            if cos_tilt < np.cos(max_tilt) and F_des[2] > 0:
                # 倾角超限且 F_z>0（机体未翻转），缩放水平分量
                s = F_des[2] * np.tan(max_tilt) / F_h
                F_des = np.array([s * F_des[0], s * F_des[1], F_des[2]])
            elif F_des[2] <= 0:
                # 极端情况: F_z ≤ 0（期望力向下），强制恢复为竖直向上
                F_des = np.array([0., 0., self.g])

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)

        # ====== 第3步：期望姿态解算 ======
        # 从推力方向 + 偏航参考 → 期望旋转矩阵 R_d
        R_d = compute_desired_attitude(b3_des, yaw_d)

        # ====== 第4步：姿态误差计算 ======
        # θ_e = Log(R·R_d')  — 指数坐标姿态误差
        # ω_e = ω - R'·R_d·ω_d — 体坐标系角速度误差
        theta_e = compute_attitude_error(R, R_d)
        omega_e = compute_omega_error(omega, omega_d, R, R_d)

        # ====== 第5步：姿态回路 ======
        # 姿态 HPC 输出：虚拟控制（角加速度指令）[rad/s²]
        u_hom = self.att_hpc.compute_virtual_control(theta_e, omega_e)

        # ====== 第6步：力矩映射 ======
        # M = J·(R_d'·u_hom - ω_d^×·ω + ω̇_d) + ω×J·ω
        M = map_virtual_to_torque(u_hom, self.J, R_d, omega, omega_d, omega_d_dot)

        # ====== 第7步：推力映射 ======
        # 关键：推力 f 沿机体 Z 轴施加力 f·R·e3 在惯性系
        # 需要 F_des_force = m·(u_pos + g·e3) [N]
        # 推力 f = F_des_force · (R·e3)（投影到当前机体 Z 轴）
        #
        # Bug修复记录（Bug #1）：
        #   之前错误地使用了 F_des（比力单位）而非 F_des_force（力单位），
        #   导致推力偏小 m 倍，位置误差发散至 10^87 米。
        F_des_force = self.m * (u_pos + np.array([0., 0., self.g]))
        thrust = np.dot(F_des_force, R @ np.array([0., 0., 1.]))

        # ====== 第8步：饱和限制 ======
        # 防止数值溢出和物理不可行的控制量
        thrust = np.clip(thrust, thrust_limits[0], thrust_limits[1])
        M = np.clip(M, -torque_limit, torque_limit)

        return np.array([thrust, M[0], M[1], M[2]])

    def get_debug_info(self, state, pos_d, vel_d, yaw_d):
        """
        返回中间信号，用于调试和日志记录。
        包含所有限幅后的实际值（与 compute_control 一致）。

        调试时可以调用此函数查看:
            - 位置误差和 HPC 输出
            - 期望力方向和大小（含倾角限制）
            - 姿态误差的指数坐标表示（使用限制后的 R_d）
        """
        from models.quadrotor_se3 import QuadrotorSE3
        model = QuadrotorSE3(self.m, self.J, self.g)
        pos, vel, R, omega = model.unpack_state(state)

        e_pos = pos - pos_d
        e_vel = vel - vel_d
        u_pos = self.pos_hpc.compute_control_vector(e_pos, e_vel)

        F_des_raw = u_pos + np.array([0., 0., self.g])
        tilt_raw_deg = np.rad2deg(np.arccos(
            np.clip(F_des_raw[2] / max(np.linalg.norm(F_des_raw), 1e-10), -1, 1)))

        # 推力倾角限制（与 compute_control 一致）
        F_des = F_des_raw.copy()
        tilt_limited = False
        F_h = np.linalg.norm(F_des[:2])
        if F_h > 1e-10:
            cos_tilt = F_des[2] / np.linalg.norm(F_des)
            if cos_tilt < np.cos(self.max_tilt) and F_des[2] > 0:
                s = F_des[2] * np.tan(self.max_tilt) / F_h
                F_des = np.array([s * F_des[0], s * F_des[1], F_des[2]])
                tilt_limited = True
            elif F_des[2] <= 0:
                F_des = np.array([0., 0., self.g])
                tilt_limited = True

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)
        tilt_deg = np.rad2deg(np.arccos(np.clip(b3_des[2], -1, 1)))

        R_d = compute_desired_attitude(b3_des, yaw_d)
        theta_e = compute_attitude_error(R, R_d)
        omega_e = omega - R.T @ R_d @ np.zeros(3)

        return {
            'e_pos': e_pos,
            'e_vel': e_vel,
            'u_pos': u_pos,
            'F_des_raw': F_des_raw,
            'F_des': F_des,
            'b3_des': b3_des,
            'R_d': R_d,
            'theta_e': theta_e,
            'omega_e': omega_e,
            'tilt_raw_deg': tilt_raw_deg,
            'tilt_deg': tilt_deg,
            'tilt_limited': tilt_limited,
        }


class LeeGeometricPD:
    """
    Lee et al. (2010) SE(3) 几何 PD 控制器 —— 对比基线

    这是文献中的经典几何跟踪控制器，用作与齐次控制器对比的基线。

    控制律 (Lee 2010, Eq. 19 & 42):
        位置回路:
            F_des = -k_x·e_pos - k_v·e_vel + m·g·e3 + m·ẍ_d
            （PD + 重力补偿 + 前馈）

        姿态回路:
            M = -k_R·θ_e - k_Ω·ω_e + ω×J·ω
                - J·(ω_d^×·R'·R_d·ω_d - R'·R_d·ω̇_d)
            （PD + 陀螺补偿 + 前馈）

    与齐次控制器的关键区别:
        1. 增益直接是"力/力矩增益"（不是"加速度增益"）
        2. 没有齐次范数的非线性缩放
        3. 收敛是指数型的（非有限时间）
    """

    def __init__(self, m, J, g=9.81,
                 kx=8.4, kv=7.0, kR=4.0, komega=2.0, max_tilt=60.0):
        """
        参数:
            m: 飞行器质量 [kg]
            J: 转动惯量矩阵
            g: 重力加速度
            kx, kv: 位置 PD 增益 [N/m] 和 [N/(m/s)]
            kR, komega: 姿态 PD 增益 [N·m/rad] 和 [N·m/(rad/s)]
            max_tilt: 最大允许推力倾角 [deg]（默认 60°）
        """
        self.m = m
        self.J = np.asarray(J)
        self.g = g
        self.kx = kx
        self.kv = kv
        self.kR = kR
        self.komega = komega
        self.max_tilt = np.deg2rad(max_tilt)

    def compute_control(self, state, pos_d, vel_d, yaw_d, acc_d=None,
                        omega_d=None, omega_d_dot=None):
        """
        计算 Lee 几何 PD 控制输入。

        处理流程（与 SE3HomogeneousController 类似但用线性 PD 替代 HPC）:
            1. 位置误差 → PD + 重力补偿 → 期望力 F_des
            2. F_des 方向 → 期望姿态 R_d
            3. 姿态误差 θ_e, ω_e → PD + 陀螺补偿 → 力矩 M
            4. F_des 投影到机体 Z 轴 → 推力
        """
        from models.quadrotor_se3 import QuadrotorSE3
        model = QuadrotorSE3(self.m, self.J, self.g)
        pos, vel, R, omega = model.unpack_state(state)

        if acc_d is None:
            acc_d = np.zeros(3)
        if omega_d is None:
            omega_d = np.zeros(3)
        if omega_d_dot is None:
            omega_d_dot = np.zeros(3)

        # 位置误差
        e_pos = pos - pos_d
        e_vel = vel - vel_d

        # 期望力 (Lee 2010, Eq. 19)
        # F_des = -kx*e - kv*ė + mg*e3 + m*ẍ_d  (单位: N)
        F_des = (-self.kx * e_pos - self.kv * e_vel +
                 self.m * self.g * np.array([0., 0., 1.]) +
                 self.m * acc_d)

        # 推力倾角限制（与 HPC 相同，默认 60°）
        max_tilt = self.max_tilt
        F_h = np.linalg.norm(F_des[:2])
        if F_h > 1e-10:
            cos_tilt = F_des[2] / np.linalg.norm(F_des)
            if cos_tilt < np.cos(max_tilt) and F_des[2] > 0:
                s = F_des[2] * np.tan(max_tilt) / F_h
                F_des = np.array([s * F_des[0], s * F_des[1], F_des[2]])
            elif F_des[2] <= 0:
                F_des = np.array([0., 0., self.m * self.g])

        b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)
        R_d = compute_desired_attitude(b3_des, yaw_d)

        # 姿态误差
        theta_e = compute_attitude_error(R, R_d)
        omega_e = omega - R.T @ R_d @ omega_d

        # 力矩 (Lee 2010, Eq. 42)
        M = (-self.kR * theta_e - self.komega * omega_e +
             np.cross(omega, self.J @ omega) -
             self.J @ (hat(omega_d) @ R.T @ R_d @ omega_d - R.T @ R_d @ omega_d_dot))

        # 推力 = F_des 投影到当前机体 Z 轴
        thrust = np.dot(F_des, R @ np.array([0., 0., 1.]))

        return np.array([thrust, M[0], M[1], M[2]])
