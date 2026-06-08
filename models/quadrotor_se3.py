"""
四旋翼 SE(3) 全动力学模型
=========================

基于 Lee, Leok & McClamroch (2010) 的 SE(3) 几何控制框架。

状态向量（18 维，R 用 3×3 矩阵展开存储）:
    state[0:3]   = [x, y, z]         位置（惯性系，Z-up）
    state[3:6]   = [vx, vy, vz]      速度（惯性系）
    state[6:15]  = R (3×3, 列主序)   旋转矩阵（机体 → 惯性系）
    state[15:18] = [ωx, ωy, ωz]     体角速度

控制输入（4 维）:
    u[0]   = thrust  总推力（沿机体 Z 轴）[N]
    u[1:4] = [τx, τy, τz]  体坐标系力矩 [N·m]

动力学方程:
    d(pos)/dt = vel
    d(vel)/dt = [0, 0, -g] + (thrust/m) * R * e3     (Newton)
    dR/dt     = R * ω^                                 (Poisson)
    dω/dt     = J⁻¹ * (τ - ω × J·ω)                  (Euler)

SO(3) 工具函数:
    hat(v) : ℝ³ → so(3)    向量 → 反对称矩阵
    vee(S) : so(3) → ℝ³    反对称矩阵 → 向量
    exp_so3(θ) : ℝ³ → SO(3)  指数映射（Rodrigues 公式）
    log_so3(R) : SO(3) → ℝ³  对数映射（指数坐标）
    jacobian_r_inv(θ) : 右 Jacobian 逆
"""

import numpy as np
from scipy.linalg import expm


def hat(v):
    """
    向量 → 反对称矩阵（hat map / skew-symmetric matrix）

    将 ℝ³ 向量映射到 so(3) Lie 代数:
        v = [v1, v2, v3]  →  v^ = [[0, -v3, v2],
                                    [v3, 0, -v1],
                                    [-v2, v1, 0]]

    用途:
        - 叉积: v × w = v^ @ w
        - 旋转运动学: Ṙ = R · ω^
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def vee(S):
    """
    反对称矩阵 → 向量（vee map，hat 的逆运算）

    从 so(3) 反对称矩阵提取 ℝ³ 向量:
        v = [S₃₂, S₁₃, S₂₁]

    满足: vee(hat(v)) = v
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class QuadrotorSE3:
    """
    四旋翼 SE(3) 刚体动力学模型

    使用 RK4 积分（每步 SVD 重投影保持 R ∈ SO(3)）。

    默认参数（来自 demo_uav/ 和 Wang Siyuan 2020）:
        m  = 1.4 kg
        J  = diag([0.0211, 0.0219, 0.0366]) kg·m²
        g  = 9.81 m/s²
    """

    def __init__(self, m=1.4, J=None, g=9.81):
        """
        参数:
            m: 质量 [kg]
            J: 转动惯量矩阵 (3×3) [kg·m²]。默认: diag([0.0211, 0.0219, 0.0366])
            g: 重力加速度 [m/s²]
        """
        self.m = m
        if J is None:
            self.J = np.diag([0.0211, 0.0219, 0.0366])
        else:
            self.J = np.asarray(J)
        self.g = g
        self.J_inv = np.linalg.inv(self.J)  # 预计算逆矩阵，避免每步求逆

    def dynamics(self, state, u):
        """
        连续时间 SE(3) 动力学（右端函数）。

        参数:
            state: 18 维状态向量
            u:     4 维控制 [thrust, τx, τy, τz]

        返回:
            dstate: 18 维状态导数
        """
        pos = state[0:3]
        vel = state[3:6]
        R = state[6:15].reshape(3, 3)   # 3×3 旋转矩阵
        omega = state[15:18]
        thrust = u[0]
        tau = u[1:4]

        # ---- 位置运动学 ----
        dpos = vel

        # ---- 速度动力学 (Newton 第二定律) ----
        # 重力: [0, 0, -g]（惯性系 Z 向上，重力向下）
        # 推力: (thrust/m) * R * e3（推力沿机体 Z 轴，旋转到惯性系）
        e3 = np.array([0., 0., 1.])
        dvel = np.array([0., 0., -self.g]) + (thrust / self.m) * (R @ e3)

        # ---- 姿态运动学 (Poisson 方程) ----
        # Ṙ = R · ω^
        dR = R @ hat(omega)

        # ---- 角速度动力学 (Euler 方程) ----
        # ω̇ = J⁻¹ · (τ - ω × J·ω)
        # ω × J·ω 是陀螺耦合项（即使 τ=0，旋转的刚体也有此项）
        domega = self.J_inv @ (tau - np.cross(omega, self.J @ omega))

        # 组装 18 维导数向量
        dstate = np.zeros(18)
        dstate[0:3] = dpos
        dstate[3:6] = dvel
        dstate[6:15] = dR.flatten()
        dstate[15:18] = domega
        return dstate

    def step_rk4(self, state, u, dt):
        """
        四阶 Runge-Kutta (RK4) 积分一步。

        每步后用 SVD 将旋转矩阵重新投影到 SO(3)，
        防止数值漂移导致 R 不再是正交矩阵。

        参数:
            state: 当前状态 (18,)
            u:     控制输入 (4,)
            dt:    积分步长 [s]

        返回:
            state_new: 下一时刻状态 (18,)
        """
        # 经典 RK4 公式
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * dt * k1, u)
        k3 = self.dynamics(state + 0.5 * dt * k2, u)
        k4 = self.dynamics(state + dt * k3, u)
        state_new = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # ---- SVD 重正交化 ----
        # 提取旋转矩阵部分，用 SVD 投影到 SO(3):
        #   R_proj = U * V'（最优 Frobenius 范数投影）
        R = state_new[6:15].reshape(3, 3)
        U, _, Vt = np.linalg.svd(R)
        R_proj = U @ Vt
        # 确保 det(R) = +1（而非 -1，后者对应反射）
        if np.linalg.det(R_proj) < 0:
            R_proj = U @ np.diag([1, 1, -1]) @ Vt
        state_new[6:15] = R_proj.flatten()
        return state_new

    def make_state(self, pos, vel, R, omega):
        """将各组件打包为 18 维状态向量。"""
        state = np.zeros(18)
        state[0:3] = np.asarray(pos).flatten()
        state[3:6] = np.asarray(vel).flatten()
        state[6:15] = np.asarray(R).flatten()
        state[15:18] = np.asarray(omega).flatten()
        return state

    def unpack_state(self, state):
        """从 18 维状态向量提取各组件。"""
        pos = state[0:3].copy()
        vel = state[3:6].copy()
        R = state[6:15].copy().reshape(3, 3)
        omega = state[15:18].copy()
        return pos, vel, R, omega


# ============================================================
# SO(3) 李群工具函数
# ============================================================

def exp_so3(theta):
    """
    指数映射: so(3) → SO(3)（Rodrigues 旋转公式）

    将指数坐标 θ = φ·n（φ=旋转角，n=旋转轴）映射为旋转矩阵:
        R = I + sin(φ)·n^ + (1-cos(φ))·(n^)²

    参数:
        theta: 指数坐标 (3,) — 方向=旋转轴，长度=旋转角 [rad]

    返回:
        R: 旋转矩阵 (3×3)
    """
    theta = np.asarray(theta).flatten()
    angle = np.linalg.norm(theta)   # φ = |θ|

    if angle < 1e-12:
        return np.eye(3)             # 零旋转 → 单位矩阵

    axis = theta / angle             # n = θ/φ（单位旋转轴）
    K = hat(axis)                    # n^（反对称矩阵）
    # Rodrigues 公式
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def log_so3(R):
    """
    对数映射: SO(3) → so(3)（指数坐标）

    将旋转矩阵映射回指数坐标:
        φ = arccos((tr(R)-1)/2)
        θ = φ/(2sin(φ)) · (R - R')∨

    值域: θ ∈ B³[π] = {θ ∈ ℝ³ : |θ| ≤ π}
    在 |θ|=π 处有歧义（±πn 对应同一个旋转矩阵）。

    参数:
        R: 旋转矩阵 (3×3)

    返回:
        theta: 指数坐标 (3,) [rad]
    """
    R = np.asarray(R).reshape(3, 3)
    # 将 trace 约束在 [-1, 3] 范围（对应 arccos 的 [-1, 1] 参数）
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    angle = np.arccos(tr)            # φ ∈ [0, π]

    if angle < 1e-12:
        return np.zeros(3)            # R ≈ I → θ ≈ 0

    if abs(angle - np.pi) < 1e-10:
        # 180° 旋转：arccos 精确但旋转轴方向有歧义
        # 从 R-I 的非零列提取旋转轴
        v = vee(R - np.eye(3))
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            return np.pi * v / v_norm
        return np.zeros(3)

    # 一般情况: θ = φ/(2sin(φ)) · (R-R')∨
    return (angle / (2 * np.sin(angle))) * vee(R - R.T)


def jacobian_r_inv(theta):
    """
    SO(3) 右 Jacobian 的逆 J_r⁻¹(θ)

    关系: dθ/dt = J_r⁻¹(θ) · ω

    即指数坐标的变化率 ≠ 角速度（除非在小角度下）。
    这是由于 SO(3) 的弯曲几何导致的非线性关系。

    公式 (Chirikjian 2012):
        J_r⁻¹ = I + ½·θ^ + (1/|θ|² - (1+cos|θ|)/(2|θ|·sin|θ|)) · (θ^)²

    参数:
        theta: 指数坐标 (3,)

    返回:
        J_r_inv: (3×3) 矩阵
    """
    theta = np.asarray(theta).flatten()
    angle = np.linalg.norm(theta)

    if angle < 1e-12:
        return np.eye(3)             # 小角度: J_r⁻¹(0) = I

    K = hat(theta)
    a = 1 / (angle ** 2)
    b = (1 + np.cos(angle)) / (2 * angle * np.sin(angle))
    return np.eye(3) + 0.5 * K + (a - b) * K @ K
