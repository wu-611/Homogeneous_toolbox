"""
姿态回路 so(3) 齐次控制器设计模块
===================================

基于 Zhou, Polyakov & Zheng (2023) "Generalized Homogeneous Rigid-Body Attitude Control"
(IEEE TAC, arXiv:2210.01530v2)

核心思想：
    不在 SO(3)（旋转矩阵）上直接定义膨胀（因为 SO(3) 是紧流形，不允许非平凡膨胀），
    而是在其 Lie 代数 so(3) ≅ ℝ³ 上定义——通过指数坐标 θ_e ∈ ℝ³ 表示姿态误差。

状态向量：
    ξ = [θ_e; ω_e] ∈ ℝ⁶
    θ_e: 指数坐标姿态误差（so(3) 元素，ℝ³ 向量）
    ω_e: 体坐标系角速度误差

关键公式（论文 Theorem 3）：
    膨胀生成元:  Gd = [(1-μ)I₃     0   ]
                      [   0        I₃  ]
                      （加权膨胀：θ_e 和 ω_e 有不同的膨胀权重）

    形状矩阵:    P  = [  I₃     εI₃  ]
                      [ εI₃   K₁⁻¹  ]
                      （ε 控制 θ_e 与 ω_e 在 Lyapunov 函数中的耦合程度）

    控制律:      u_hom = ||ξ||_d^(1+μ) · K · expm(-ln||ξ||_d · Gd) · ξ
                K = [-K₁  -k₂I₃]    （类 PD 结构）

    力矩映射:    M = J·(R_d'·u_hom - ω_d^×·ω + ω̇_d) + ω×J·ω    （Eq. 23）

重要注意事项：
    u_hom 是"角加速度" [rad/s²]，不是力矩 [N·m]
    实际力矩 M = J * u_hom（近似，忽略陀螺项）
    所以有效力矩增益 ≈ J * K1（比 K1 小约 J ≈ 0.02 倍）
"""

import numpy as np
from scipy.linalg import expm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hcs_toolbox_py import hnorm


class AttitudeHPC:
    """
    so(3) 指数坐标上的姿态齐次控制器

    控制律:
        u_hom = ||ξ||_d^(1+μ) * K * expm(-ln||ξ||_d * Gd) * ξ

    其中:
        ξ = [θ_e; ω_e]  — 6维状态向量
        K = [-K₁, -k₂I₃] — 增益矩阵（类 PD）
        Gd = block_diag((1-μ)I₃, I₃) — 膨胀生成元

    μ=0 时退化为指数坐标上的几何 PD 控制器:
        u_hom = -K₁·θ_e - k₂·ω_e
    """

    def __init__(self, J, K1=None, k2=None, mu=0.0, eps=None):
        """
        参数:
            J: 转动惯量矩阵 (3×3) [kg·m²]
            K1: 比例增益。若为标量则 K1=k1·I₃。
                默认 200（有效力矩增益 = J·K1 ≈ 4 N·m/rad）
                注意：这是"角加速度增益" [rad/s² per rad]
            k2: 阻尼增益。默认 100。
                注意：这是"角加速度阻尼" [rad/s² per rad/s]
            mu: 齐次度
                mu=0:  指数收敛（退化为几何 PD）
                mu<0:  有限时间收敛
                mu>0:  近固定时间收敛
            eps: P 矩阵耦合参数。None 则自动计算。
        """
        self.J = np.asarray(J)
        self.J_inv = np.linalg.inv(self.J)
        self.mu = mu
        self.n = 6          # 状态维数（R⁶）

        # ============ 增益设计 ============
        # 关键区别：u_hom 是角加速度 [rad/s²]，不是力矩 [N·m]
        # 力矩 M = J * u_hom，所以有效力矩增益 = J * K1
        # 对于 J ≈ 0.02 的旋翼：
        #   K1=200 → 有效力矩 ≈ 4 N·m/rad（与 Lee PD 的 kR=4 可比）
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

        # 完整的 3×6 增益矩阵 K = [-K₁, -k₂I₃]
        self.K = np.hstack([-self.K1, -self.k2 * np.eye(3)])

        # ============ 膨胀生成元 Gd (Zhou 2023, Eq. 21a) ============
        # Gd = [(1-μ)I₃, 0; 0, I₃]
        # 当 μ<0 时，(1-μ)>1 → θ_e 的膨胀权重大于 ω_e
        # 含义：姿态误差比角速度误差需要更大的"压缩"力度
        self.Gd = np.zeros((6, 6))
        self.Gd[0:3, 0:3] = (1 - mu) * np.eye(3)    # θ_e 通道：权重 1-μ
        self.Gd[3:6, 3:6] = np.eye(3)                 # ω_e 通道：权重 1

        # ============ 形状矩阵 P (Zhou 2023, Eq. 21b) ============
        # P = [I₃, εI₃; εI₃, K₁⁻¹]
        # ε 需要同时满足 P≻0 和 PGd+Gd'P≻0
        if eps is None:
            eps = self._compute_eps()
        self.eps = eps

        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = np.eye(3)                   # 左上：I₃
        self.P[0:3, 3:6] = eps * np.eye(3)             # 右上：εI₃
        self.P[3:6, 0:3] = eps * np.eye(3)             # 左下：εI₃
        self.P[3:6, 3:6] = np.linalg.inv(self.K1)      # 右下：K₁⁻¹

        # 验证 P 的正定性和 PGd+Gd'P 的正定性
        self._validate()

    def _compute_eps(self):
        """
        自动计算 ε 的容许值。

        约束1: P ≻ 0 要求 ε² < λ_min(K₁⁻¹)
               → ε < √(1/λ_max(K₁)) = √(λ_min(K₁⁻¹))

        约束2: PGd + Gd'P ≻ 0 要求
               ε < 2√(1-μ) / ((2-μ)·√λ_max(K₁))
               （从块矩阵的 Schur 补条件推导）

        取 ε = 0.5·min(ε_pd, ε_gd) 留安全裕度。
        """
        K1_inv = np.linalg.inv(self.K1)

        # 约束1：P 正定
        eps_pd = np.sqrt(np.min(np.linalg.eigvals(K1_inv)))

        # 约束2：膨胀单调性 PGd+Gd'P ≻ 0
        lambda_max = np.max(np.linalg.eigvals(self.K1))
        if abs(self.mu - 1.0) < 1e-10:
            eps_gd = np.inf       # μ→1 时，约束自动满足
        else:
            eps_gd = (2.0 * np.sqrt(1.0 - self.mu) /
                      ((2.0 - self.mu) * np.sqrt(lambda_max)))

        eps = 0.5 * min(eps_pd, eps_gd)
        return max(eps, 1e-6)     # 保证至少有一个极小正值

    def _validate(self):
        """检查 P ≻ 0 和 PGd+Gd'P ≻ 0"""
        eig_P = np.linalg.eigvals(self.P)
        if np.any(eig_P <= 0):
            print(f"Warning: P 有非正特征值: {eig_P}")

        M = self.P @ self.Gd + self.Gd.T @ self.P
        eig_M = np.linalg.eigvals(M)
        if np.any(eig_M <= 0):
            print(f"Warning: PGd+Gd'P 有非正特征值: {eig_M}")

    def homogeneous_norm(self, xi):
        """
        计算状态 ξ 的典范齐次范数。

        ||ξ||_d = exp(c),  c 满足:
            ξ' · d'(-c) · P · d(-c) · ξ = 1

        其中 d(s) = expm(s·Gd) 是膨胀算子。

        计算方法：二分法求解 c，精度 1e-6。
        """
        return hnorm(xi, self.Gd, self.P)

    def compute_virtual_control(self, theta_e, omega_e):
        """
        计算虚拟控制 u_hom（角加速度误差指令）。

        参数:
            theta_e: 指数坐标姿态误差 (3维) [rad]
            omega_e: 角速度误差 (3维) [rad/s]

        返回:
            u_hom: 齐次控制量 (3维) [rad/s²] — 角加速度指令

        公式:
            u_hom = ||ξ||_d^(1+μ) · K · expm(-ln||ξ||_d · Gd) · ξ

        实现细节:
            1. 拼合状态 ξ = [θ_e; ω_e]
            2. 计算齐次范数 nx = ||ξ||_d
            3. 对 nx 施加饱和 [alpha, beta] 防止数值问题
            4. 计算齐次投影 d(-ln(nx))·ξ = expm(-ln(nx)·Gd)·ξ
            5. 输出 u_hom = nx^(1+μ) · K · (齐次投影)
        """
        xi = np.concatenate([theta_e, omega_e])

        # 零状态：无需控制
        if np.linalg.norm(xi) < 1e-12:
            return np.zeros(3)

        # 计算齐次范数
        nx = self.homogeneous_norm(xi)

        # 饱和处理：
        #   alpha=0.1: 防止 ||ξ||_d 过小 → 1/||ξ||_d 爆炸 → 数值溢出
        #   beta=10.0: 防止 ||ξ||_d 过大 → ||ξ||_d^(1+μ) 溢出
        alpha = 0.1
        beta = 10.0
        nx_sat = max(alpha, min(beta, nx))

        if nx_sat < 1e-20:
            return np.zeros(3)

        # 齐次投影: 将状态投影到单位球面上
        # d(-ln(nx)) = expm(-ln(nx)·Gd)
        # 投影后的向量满足 ||d(-ln(nx))·ξ||_d = 1
        d_proj = expm(-np.log(nx_sat) * self.Gd) @ xi

        # 控制律: u = nx^(1+μ) * K * d_proj
        # K = [-K1, -k2*I3], 所以 K @ d_proj = -K1*θ_proj - k2*ω_proj
        u_hom = (nx_sat ** (1.0 + self.mu)) * (self.K @ d_proj)
        return u_hom

    def compute_torque(self, theta_e, omega_e, R_d, omega, omega_d, omega_d_dot):
        """
        从虚拟控制计算实际体坐标系力矩。

        实现力矩映射 (Zhou 2023, Eq. 23):
            M = J · (R_d' · u_hom - ω_d^× · ω + ω̇_d) + ω × J·ω

        各项物理含义:
            J·R_d'·u_hom    : 将 so(3) 齐次控制 → 惯性系力矩 → 体坐标系
            -J·ω_d^×·ω      : 补偿期望角速度与实际角速度的耦合
            J·ω̇_d           : 前馈期望角加速度
            ω×J·ω           : 陀螺效应补偿（欧拉力/科里奥利力）

        参数:
            theta_e:     姿态误差（指数坐标）[rad]
            omega_e:     角速度误差 [rad/s]
            R_d:         期望姿态矩阵 (3×3)
            omega:       当前体角速度 [rad/s]
            omega_d:     期望体角速度 [rad/s]
            omega_d_dot: 期望体角加速度 [rad/s²]

        返回:
            M: 体坐标系力矩指令 [tau_x, tau_y, tau_z] [N·m]
        """
        u_hom = self.compute_virtual_control(theta_e, omega_e)

        from models.quadrotor_se3 import hat
        M = (self.J @ (R_d.T @ u_hom - np.cross(omega_d, omega) + omega_d_dot) +
             np.cross(omega, self.J @ omega))
        return M
