#!/usr/bin/env python3
"""
悬停调试脚本 —— 用于快速验证控制器在简单场景下的行为
=====================================================

功能:
    - 测试 Lee PD 和 HPC 在小偏移悬停时的控制输出
    - 打印中间变量用于手动检查（位置误差、期望力、姿态误差、力矩）
    - 验证推力是否接近悬停值 (m·g ≈ 13.7 N)

使用方法:
    python3 scripts/test_hover.py

预期输出:
    两个控制器都应产生接近 m·g 的推力和接近零的力矩。
    如果 HPC 推力显著偏离 m·g，说明参数配置有问题。
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.quadrotor_se3 import QuadrotorSE3
from controllers.se3_homogeneous_full import SE3HomogeneousController, LeeGeometricPD

# ---- 四旋翼参数 ----
m = 1.4
J = np.diag([0.0211, 0.0219, 0.0366])
g = 9.81
model = QuadrotorSE3(m, J, g)
dt = 0.001

# ---- 测试场景：悬停在原点 ----
pos_d = np.array([0.0, 0.0, 0.0])
vel_d = np.zeros(3)
yaw_d = 0.0

# 小偏移初始状态
pos0 = np.array([0.1, 0.0, -0.1])  # 10cm 位置误差
vel0 = np.zeros(3)
R0 = np.eye(3)                      # 初始姿态水平
omega0 = np.zeros(3)

# ====== 测试 1: Lee PD ======
print("=" * 65)
print("悬停调试 — Lee PD vs HPC 对比")
print("=" * 65)

print("\n>>> Lee PD 悬停测试 (0.2s)")
state = model.make_state(pos0, vel0, R0, omega0)
lee = LeeGeometricPD(m, J, g, kx=8.4, kv=7.0, kR=4.0, komega=2.0)
for k in range(200):  # 0.2 秒
    u = lee.compute_control(state, pos_d, vel_d, yaw_d)
    state = model.step_rk4(state, u, dt)
pos, vel, R, omega = model.unpack_state(state)
print(f"  终态位置: {pos}")
print(f"  终态速度: {vel}")
print(f"  控制输入: thrust={u[0]:.2f} N, τ={u[1:4]} N·m")
print(f"  (悬停推力理论值 = m·g = {m*g:.2f} N)")

# ====== 测试 2: HPC μ=0 ======
print("\n>>> HPC μ=0 悬停测试 (0.2s)")
state = model.make_state(pos0, vel0, R0, omega0)
hpc = SE3HomogeneousController(m, J, g, mu_p=0.0, mu_a=0.0)
for k in range(200):
    u = hpc.compute_control(state, pos_d, vel_d, yaw_d)
    state = model.step_rk4(state, u, dt)
pos, vel, R, omega = model.unpack_state(state)
print(f"  终态位置: {pos}")
print(f"  终态速度: {vel}")
print(f"  控制输入: thrust={u[0]:.2f} N, τ={u[1:4]} N·m")

# ====== 测试 3: 调试信息（单步） ======
print("\n>>> HPC 中间变量（初始时刻，单步）")
state = model.make_state(pos0, vel0, R0, omega0)
debug = hpc.get_debug_info(state, pos_d, vel_d, yaw_d)
print(f"  位置误差 e_pos:  {debug['e_pos']}")
print(f"  速度误差 e_vel:  {debug['e_vel']}")
print(f"  HPC 输出 u_pos:  {debug['u_pos']} [m/s²]")
print(f"  期望力方向 b3:   {debug['b3_des']}")
print(f"  姿态误差 θ_e:    {debug['theta_e']} [rad]")

# 单步控制输出
u_hpc = hpc.compute_control(state, pos_d, vel_d, yaw_d)
u_lee = lee.compute_control(state, pos_d, vel_d, yaw_d)
print(f"\n  单步对比:")
print(f"    Lee PD:  thrust={u_lee[0]:.4f}, τ_y={u_lee[2]:.4f}")
print(f"    HPC μ=0: thrust={u_hpc[0]:.4f}, τ_y={u_hpc[2]:.4f}")
print(f"    推力比 HPC/Lee: {u_hpc[0]/u_lee[0]:.4f} (应接近 1.0)")
print(f"    力矩比 HPC/Lee: {u_hpc[2]/u_lee[2]:.4f} (应接近 1.0)")

print("\n完成!")
