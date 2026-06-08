"""
力矩映射模块：将虚拟 so(3) 控制转换为物理体坐标系力矩
=====================================================

实现 Zhou 2023 Eq. 23 的逆动力学映射:
    M = J · (R_d' · u_hom - ω_d^× · ω + ω̇_d) + ω × J·ω

以及推力映射:
    F = m · (a_z + g)  （从期望 Z 加速度计算推力）

物理背景：
    齐次姿态控制器输出的是"虚拟控制" u_hom = ω̇_e（角加速度误差）。
    但实际飞行器的控制输入是力矩 M（体坐标系，单位 N·m）。
    这个模块通过刚体逆动力学将 u_hom 映射为 M。

推导过程（详见 齐次控制从积分链到SO3的详细扩展.md 第14节）：
    1. u_hom = ω̇_e  （虚拟控制 = 误差角加速度）
    2. ω_e = R_d · (ω - ω_d)  →  ω̇_e = R_d·ω_d^×·(ω-ω_d) + R_d·(ω̇-ω̇_d)
    3. ω̇ = J⁻¹·(-ω×Jω + M)   （Euler 方程）
    4. 代入并解出 M →  得到 Eq. 23
"""

import numpy as np
from models.quadrotor_se3 import hat


def map_virtual_to_torque(u_hom, J, R_d, omega, omega_d, omega_d_dot):
    """
    将虚拟齐次控制 u_hom 转换为实际体坐标系力矩 M。

    公式 (Zhou 2023, Eq. 23):
        M = J · (R_d' · u_hom - ω_d^× · ω + ω̇_d) + ω × J·ω

    各项物理含义:
        J·R_d'·u_hom    : 齐次控制从 so(3) → 惯性系力矩 → 体坐标系
                          先乘以 R_d' 将 u_hom 从"期望体坐标系"转到惯性系，
                          再乘以 J 将角加速度转为力矩
        -J·ω_d^×·ω      : 补偿期望角速度耦合项
                          ω_d^×·ω = ω_d × ω（叉积）
                          来自运动学推导中的 R_d·ω_d^×·(ω-ω_d) 项
        J·ω̇_d           : 前馈项——如果期望角速度在变化，需要额外力矩
                          来产生对应的角加速度
        ω×J·ω           : 欧拉力/科里奥利力补偿（刚体陀螺效应）
                          即使没有外力矩，旋转的刚体也会有此交叉项

    参数:
        u_hom:       虚拟齐次控制量 [rad/s²] — 期望误差角加速度
        J:           转动惯量矩阵 (3×3) [kg·m²]
        R_d:         期望姿态矩阵 (3×3)
        omega:       当前体角速度 [rad/s]
        omega_d:     期望体角速度 [rad/s]
        omega_d_dot: 期望体角加速度 [rad/s²]

    返回:
        M: 体坐标系力矩 [τ_x, τ_y, τ_z] [N·m]
    """
    return (J @ (R_d.T @ u_hom - np.cross(omega_d, omega) + omega_d_dot) +
            np.cross(omega, J @ omega))


def map_accel_to_thrust(u_pos_z, m, g=9.81):
    """
    将 Z 轴加速度指令转换为总推力。

    公式: F = m · (a_z + g)

    物理含义:
        在惯性系中，净加速度 = (推力/m) - g
        所以 推力 = m · (目标加速度 + g)
        这是在"机体竖直"假设下的简化公式。

        注意：这个函数假设机体 Z 轴与惯性系 Z 轴对齐（水平悬停）。
        一般情况下应使用推力投影公式：
            thrust = F_des_force · (R · e3)
        见 se3_homogeneous_full.py 中的完整实现。

    参数:
        u_pos_z: Z 轴期望加速度 [m/s²]（来自位置控制器）
        m:       飞行器质量 [kg]
        g:       重力加速度 [m/s²]

    返回:
        thrust: 总推力 [N]
    """
    return m * (u_pos_z + g)


def compute_control_input(u_pos, u_hom, m, g, J, R, R_d, omega, omega_d,
                          omega_d_dot):
    """
    完整的控制映射：位置 + 姿态虚拟控制 → [推力, 力矩]。

    这是高层接口函数，封装了推力映射和力矩映射。

    参数:
        u_pos:       位置 HPC 输出的期望加速度 (3维) [m/s²]
        u_hom:       姿态 HPC 输出的虚拟控制 (3维) [rad/s²]
        m, g:        质量和重力加速度
        J:           转动惯量矩阵
        R, R_d:      当前和期望旋转矩阵
        omega:       当前体角速度
        omega_d:     期望体角速度
        omega_d_dot: 期望体角加速度

    返回:
        u: [thrust, τ_x, τ_y, τ_z] — 四旋翼 4 通道控制输入
    """
    thrust = map_accel_to_thrust(u_pos[2], m, g)
    M = map_virtual_to_torque(u_hom, J, R_d, omega, omega_d, omega_d_dot)
    return np.array([thrust, M[0], M[1], M[2]])
