"""
期望姿态解算模块
================

基于 Lee, Leok & McClamroch (2010) "Geometric Tracking Control of a Quadrotor UAV on SE(3)"
(CDC 2010)

功能：从位置控制器输出的期望推力方向，解算出完整的期望姿态 (R_d, ω_d, ω̇_d)。

核心问题：
    四旋翼是欠驱动系统——4 个控制输入（总推力 + 3 轴力矩），6 个自由度。
    位置控制不直接产生水平力，而是通过"倾斜机体"来产生水平推力分量。

    位置控制器 → 期望加速度 a_des → 期望力方向 b3_des → 期望姿态 R_d
                                             ↑
                                         这个模块做的事

坐标系约定（惯性系 Z-up）：
    b3_des: 机体 Z 轴在惯性系中的期望方向（= 推力方向）
    b1_des: 机体 X 轴在惯性系中的期望方向（由偏航参考决定）
    b2_des: 机体 Y 轴 = b3_des × b1_des
    R_d = [b1_des, b2_des, b3_des]  (3×3 旋转矩阵，列向量为机体轴)
"""

import numpy as np
from models.quadrotor_se3 import hat


def compute_desired_attitude(b3_des, yaw_des=0.0):
    """
    从期望推力方向和偏航参考计算期望旋转矩阵 R_d。

    算法:
        1. b3_des: 期望推力方向（归一化）
        2. b1_ref = [cos(ψ), sin(ψ), 0]: 偏航参考方向（水平面投影）
        3. b2_des = b3_des × b1_ref / |...|  （垂直于推力方向和偏航参考）
        4. b1_des = b2_des × b3_des            （构成右手系）
        5. R_d = [b1_des, b2_des, b3_des]

    奇点处理:
        当 b3_des ∥ b1_ref 时（推力方向纯竖直），叉积为零。
        此时使用替代参考方向 b1_ref' = [0, cos(ψ), sin(ψ)]。

    参数:
        b3_des: 期望推力方向（单位向量，惯性系）(3,)
        yaw_des: 期望偏航角 [rad]

    返回:
        R_d: 期望旋转矩阵 (3×3)，列向量为机体轴在惯性系中的方向
    """
    b3 = np.asarray(b3_des).flatten()
    b3 = b3 / np.linalg.norm(b3)  # 确保是单位向量

    # b1_ref: 偏航参考在水平面的投影方向
    b1_ref = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])

    # b2_des = b3_des × b1_ref（未归一化）
    b2 = np.cross(b3, b1_ref)
    b2_norm = np.linalg.norm(b2)

    # 奇点检测：b3_des 与 b1_ref 近似平行
    if b2_norm < 1e-10:
        # 使用替代参考方向（在 YZ 平面内）
        b1_ref = np.array([0., np.cos(yaw_des), np.sin(yaw_des)])
        b2 = np.cross(b3, b1_ref)
        b2_norm = np.linalg.norm(b2)

    b2 = b2 / b2_norm  # 归一化 → 机体 Y 轴方向

    # b1_des = b2_des × b3_des（右手定则）
    b1 = np.cross(b2, b3)

    # 组装旋转矩阵：列向量 = 机体轴在惯性系的方向
    R_d = np.column_stack([b1, b2, b3])
    return R_d


def compute_desired_angular_velocity(R_d, R_d_dot):
    """
    从 R_d 及其导数计算期望体角速度 ω_d。

    公式: ω_d = (R_d' · Ṙ_d)∨   （∨ 表示从反对称矩阵提取向量）

    这是旋转运动学 Ṙ = R·ω^ 的逆运算：
        如果 Ṙ_d = R_d · ω_d^
        则 ω_d^ = R_d' · Ṙ_d
        则 ω_d = (R_d' · Ṙ_d)∨

    参数:
        R_d: 期望旋转矩阵 (3×3)
        R_d_dot: 期望旋转矩阵的时间导数 (3×3)

    返回:
        omega_d: 期望体角速度 [rad/s]
    """
    from models.quadrotor_se3 import vee
    return vee(R_d.T @ R_d_dot)


def compute_attitude_error(R, R_d):
    """
    计算指数坐标姿态误差。

    公式:
        R_e = R · R_d'           （姿态误差旋转矩阵：从期望到当前）
        θ_e = Log(R_e)           （对数映射 → 指数坐标）

    θ_e 的物理含义:
        |θ_e| = 旋转角（最小转角，∈ [0, π]）
        θ_e/|θ_e| = 旋转轴（单位向量）

    参数:
        R: 当前旋转矩阵 (3×3)
        R_d: 期望旋转矩阵 (3×3)

    返回:
        theta_e: 指数坐标姿态误差 [rad]（3维向量）
    """
    from models.quadrotor_se3 import log_so3
    R_e = R @ R_d.T
    return log_so3(R_e)


def compute_omega_error(omega, omega_d, R, R_d):
    """
    计算体坐标系下的角速度误差。

    公式:
        ω_e = ω - R'·R_d·ω_d

    含义:
        当前角速度 ω 与（映射到当前体坐标系的）期望角速度之差。
        先把期望角速度从"期望体坐标系"旋转到惯性系（乘以 R_d），
        再旋转到"当前体坐标系"（乘以 R'）。

    参数:
        omega:   当前体角速度 [rad/s]
        omega_d: 期望体角速度 [rad/s]
        R:       当前旋转矩阵
        R_d:     期望旋转矩阵

    返回:
        omega_e: 角速度误差 [rad/s]
    """
    return omega - R.T @ R_d @ omega_d
