"""
位置回路齐次比例控制器 (Position HPC) 设计模块
=================================================

功能：将四旋翼位置动力学的线性反馈增益升级为齐次控制器。

数学模型：
    每个通道 (x, y, z) 是独立的双积分器：
        A_i = [[0, 1], [0, 0]]      # 位置→速度→加速度 的积分链
        B_i = [[0], [1/m]]           # 控制输入是加速度 [m/s²]

    线性控制律：u = K * [e_pos; e_vel]    (加速度指令)
    齐次升级：  通过 lpc2hpc(A, B, K) 获得 K0, G0, P, μ范围

关键概念：
    - μ = 0:   Gd = I（均匀膨胀），齐次范数退化为加权欧氏范数，
               控制律退化为线性反馈 u = K*x
    - μ < 0:   Gd = I + μ*G0（非均匀膨胀），
               有限时间收敛保证
    - 输出的是"期望加速度" [m/s²]，不是力 [N]
      （力的转换在 se3_homogeneous_full.py 中完成）
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hcs_toolbox_py import lpc2hpc, hnorm


class PositionHPC:
    """
    位置回路齐次比例控制器

    三个独立通道 (x, y, z)，每个是双积分器。
    输出：惯性系下的期望加速度 (F_des / m) [m/s²]。
    """

    def __init__(self, m, K_linear=None, mu=0.0):
        """
        参数:
            m: 飞行器质量 [kg]
            K_linear: 线性反馈增益，格式 [k_p, k_d]（位置和速度增益）
                      默认：极点配置在 s=-2, -3 → K = [-6m, -5m]
            mu: 齐次度
                mu=0: 指数收敛（均匀膨胀，退化为线性反馈）
                mu<0: 有限时间收敛（非均匀膨胀）
        """
        self.m = m
        self.mu = mu
        self.g = 9.81

        # ---- 每个通道的双积分器模型 ----
        self.A = np.array([[0., 1.], [0., 0.]])   # 状态矩阵
        self.B = np.array([[0.], [1. / m]])        # 控制矩阵（注意含 1/m）

        # ---- 线性增益 ----
        if K_linear is None:
            # 默认：期望闭环极点 -2, -3
            # 特征方程: s² + 5s + 6 = 0  →  k_p/m=6, k_d/m=5
            # K = [-6m, -5m] = [-8.4, -7.0]（对 m=1.4）
            self.K_linear = np.array([-6.0 * m, -5.0 * m])

        # ---- 升级为齐次控制器 ----
        self._design()

    def _design(self):
        """
        核心升级步骤：
        1. 调用 lpc2hpc 获得齐次化参数
        2. 检查 μ 是否在容许范围内
        3. 构造膨胀生成元 Gd = I + μ*G0
        4. 构造齐次范数函数 hn_fun(x) = ||x||_d
        """
        # 步骤1：lpc2hpc 升级
        # 输入: A(2×2), B(2×1), K(1×2)
        # 输出: K0(1×2) — 线性分量增益
        #       G0(2×2) — 膨胀生成元基础矩阵
        #       P(2×2)  — 形状矩阵（定义齐次范数的加权欧氏范数）
        #       mu_min, mu_max — μ 的容许范围
        K0, G0, P, mu_min, mu_max = lpc2hpc(self.A, self.B,
                                             self.K_linear.reshape(1, -1))

        self.K0 = K0
        self.G0 = G0
        self.P = P
        self.mu_min = mu_min
        self.mu_max = mu_max

        # 步骤2：检查 μ 是否在 [mu_min, mu_max] 内
        mu = np.clip(self.mu, self.mu_min + 1e-6, self.mu_max - 1e-6)
        if abs(mu - self.mu) > 1e-10:
            print(f"Warning: mu={self.mu} 超出容许范围 [{self.mu_min:.4f}, "
                  f"{self.mu_max:.4f}]，已裁剪为 {mu:.4f}")
        self.mu = mu

        # 步骤3：构造膨胀生成元
        # Gd = I + μ*G0
        # μ=0 时 Gd=I（均匀膨胀），μ≠0 时 Gd≠I（非均匀/加权膨胀）
        if abs(self.mu) < 1e-10:
            self.Gd = np.eye(2)        # 退化为均匀膨胀
        else:
            self.Gd = np.eye(2) + self.mu * self.G0  # 非均匀膨胀

        # 步骤4：非线性增益 K_nl = K_linear - K0
        # 齐次控制律分解为: u = K0*x + ||x||_d^(1+μ) * K_nl * d(-ln||x||_d) * x
        self.K_nl = self.K_linear.reshape(1, -1) - self.K0

        # 步骤5：齐次范数函数（闭包捕获 Gd 和 P）
        # hnorm(x, Gd, P) 通过二分法求解:
        #   x' * d'(-ln||x||_d) * P * d(-ln||x||_d) * x = 1
        self._hn_fun = lambda x: hnorm(x, self.Gd, self.P)

    def compute_control(self, e_pos, e_vel):
        """
        计算单个通道的期望加速度。

        参数:
            e_pos: 位置误差（标量）[m]
            e_vel: 速度误差（标量）[m/s]

        返回:
            u_pos: 期望加速度 [m/s²]

        齐次控制律:
            u = K0*x + ||x||_d^(1+μ) * K_nl * expm(-ln||x||_d * Gd) * x

        物理含义:
            - 当 ||x||_d > 1（远离平衡点）且 μ<0 时：
              ||x||_d^(1+μ) < 1 → 衰减控制，避免超调
            - 当 ||x||_d < 1（接近平衡点）且 μ<0 时：
              ||x||_d^(1+μ) > 1 → 放大控制，加速收敛
        """
        from hcs_toolbox_py import e_hpc

        x = np.array([e_pos, e_vel])
        # e_hpc 实现 u = K0*x + nx^(1+μ) * K_nl * expm(-ln(nx)*Gd) * x
        # alpha=0.1: 齐次范数下界（防止近零点增益过大 → 颤振）
        # beta=10.0: 齐次范数上界（防止远点增益过小 → 控制不足）
        u, = e_hpc(x, self.K0, self.K_nl, self.Gd, self.mu, self._hn_fun,
                   alpha=0.1, beta=10.0)
        return u

    def compute_control_vector(self, e_pos_vec, e_vel_vec):
        """
        计算三个通道（x, y, z）的期望加速度向量。

        参数:
            e_pos_vec: 位置误差 [e_x, e_y, e_z] [m]
            e_vel_vec: 速度误差 [e_vx, e_vy, e_vz] [m/s]

        返回:
            u_pos: 期望加速度向量 [a_x, a_y, a_z] [m/s²]
                   （在惯性系 Z-up 坐标系下）
        """
        return np.array([
            self.compute_control(e_pos_vec[0], e_vel_vec[0]),
            self.compute_control(e_pos_vec[1], e_vel_vec[1]),
            self.compute_control(e_pos_vec[2], e_vel_vec[2])
        ])

    def get_desired_thrust_direction(self, e_pos_vec, e_vel_vec):
        """
        从位置控制器输出计算期望推力方向。

        参数:
            e_pos_vec, e_vel_vec: 位置和速度误差

        返回:
            F_des:    总期望力向量（含重力补偿）[m/s² 比力单位]
            b3_des:   期望推力方向（机体 Z 轴在惯性系中的单位向量）
            thrust_magnitude: 期望推力大小（比力单位）
        """
        u_pos = self.compute_control_vector(e_pos_vec, e_vel_vec)
        # F_des = 期望加速度 + 重力补偿
        # 重力方向: [0, 0, -g]（惯性系 Z 向上），需要 +g 抵消
        F_des = u_pos + np.array([0., 0., self.g])
        thrust_magnitude = np.linalg.norm(F_des)

        if thrust_magnitude < 1e-10:
            b3_des = np.array([0., 0., 1.])  # 默认竖直向上
        else:
            b3_des = F_des / thrust_magnitude

        return F_des, b3_des, thrust_magnitude
