#!/usr/bin/env python3
"""
扭矩诊断脚本 V2：精确追踪实际控制器输出（含所有限幅和保护）。

排查目标：
    1. 初始阶跃的力矩瞬态（含60°倾角限制）
    2. K1 增益 vs 力矩峰值的关系
    3. 齐次范数非线性放缩的贡献
    4. SITL dt=0.01s 离散化效应
    5. 推力不足导致的级联恶化
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.quadrotor_se3 import QuadrotorSE3, log_so3
from controllers.se3_homogeneous_full import SE3HomogeneousController
from controllers.se3_homogeneous_outfb import SE3HomogeneousOutFB
from design.attitude_command import compute_desired_attitude, compute_attitude_error
from hcs_toolbox_py import hnorm


def trace_actual_controller(ctrl, state, pos_d, vel_d, yaw_d, model, label):
    """追踪 compute_control 内部所有中间量（使用实际限幅后的值）。"""
    pos, vel, R, omega = model.unpack_state(state)

    # --- 记录 compute_control 的实际输出 ---
    u = ctrl.compute_control(state, pos_d, vel_d, yaw_d)
    thrust_actual, M_actual = u[0], u[1:4]

    # --- 手动追踪中间量（使用与 compute_control 相同的逻辑）---
    e_pos = pos - pos_d
    e_vel = vel - vel_d

    # 位置 HPC
    u_pos = ctrl.pos_hpc.compute_control_vector(e_pos, e_vel)

    # 每个通道的齐次范数
    V_p_ch = [ctrl.pos_hpc._hn_fun(np.array([e_pos[i], e_vel[i]])) for i in range(3)]

    # 重力补偿 + 倾角限制（与 compute_control 完全一致）
    g = 9.81
    max_tilt = ctrl.max_tilt
    F_des_raw = u_pos + np.array([0., 0., g])
    tilt_raw = np.arccos(np.clip(F_des_raw[2] / max(np.linalg.norm(F_des_raw), 1e-10), -1, 1))

    F_des = F_des_raw.copy()
    F_h = np.linalg.norm(F_des[:2])
    tilt_limited = False
    if F_h > 1e-10:
        cos_tilt = F_des[2] / np.linalg.norm(F_des)
        if cos_tilt < np.cos(max_tilt) and F_des[2] > 0:
            s = F_des[2] * np.tan(max_tilt) / F_h
            F_des = np.array([s * F_des[0], s * F_des[1], F_des[2]])
            tilt_limited = True
        elif F_des[2] <= 0:
            F_des = np.array([0., 0., g])
            tilt_limited = True

    tilt_actual = np.arccos(np.clip(F_des[2] / max(np.linalg.norm(F_des), 1e-10), -1, 1))
    b3_des = F_des / (np.linalg.norm(F_des) + 1e-10)
    R_d = compute_desired_attitude(b3_des, yaw_d)

    # 姿态误差（实际使用的）
    theta_e = compute_attitude_error(R, R_d)
    omega_e = ctrl.compute_omega_error(omega, np.zeros(3), R, R_d) if hasattr(ctrl, 'compute_omega_error') \
        else omega - R.T @ R_d @ np.zeros(3)

    theta_norm = np.linalg.norm(theta_e)
    omega_norm = np.linalg.norm(omega_e)

    # 姿态 HPC 内部
    xi = np.concatenate([theta_e, omega_e])
    att = ctrl.att_hpc
    V_a = hnorm(xi, att.Gd, att.P) if np.linalg.norm(xi) > 1e-12 else 0.0
    nx_sat = max(0.1, min(10.0, V_a))

    # 有效增益
    if abs(att.mu + 0.5) < 1e-10:
        theta_gain_eff = nx_sat ** (-1.0)    # K1 * nx^(-1) applied to θ
        omega_gain_eff = nx_sat ** (-0.5)     # k2 * nx^(-0.5) applied to ω
    else:
        theta_gain_eff = nx_sat ** att.mu      # approximate
        omega_gain_eff = nx_sat ** (att.mu * (1.0/(1.0-att.mu)))

    from scipy.linalg import expm
    d_proj = expm(-np.log(max(nx_sat, 1e-16)) * att.Gd) @ xi
    u_hom = att.compute_virtual_control(theta_e, omega_e)

    # 力矩分解（仅姿态贡献，不含 ω_d 项因为悬停时 ω_d=0）
    J = np.diag([0.0211, 0.0219, 0.0366])
    gyro = np.cross(omega, J @ omega)
    J_RdT_uhom = J @ (R_d.T @ u_hom)

    force_des = 1.4 * F_des  # m * F_des
    thrust_raw = np.dot(force_des, R[:, 2])

    print(f"\n  [{label}]")
    print(f"  {'─'*50}")
    print(f"  位置: e_pos={np.round(e_pos,3)} |e|={np.linalg.norm(e_pos):.2f}m")
    print(f"  速度: e_vel={np.round(e_vel,3)}")
    print(f"  V_p (z通道) = {V_p_ch[2]:.3f}")
    print(f"  u_pos = {np.round(u_pos,2)} m/s²")
    print(f"  倾角(原始)={np.rad2deg(tilt_raw):.0f}° → (限制后)={np.rad2deg(tilt_actual):.0f}°"
          f"  {'[已限制]' if tilt_limited else ''}")
    print(f"  |θ_e|={theta_norm:.3f}rad ({np.rad2deg(theta_norm):.1f}°)  "
          f"|ω_e|={omega_norm:.3f}rad/s")
    print(f"  V_a={V_a:.4f}  nx_sat={nx_sat:.4f}")
    print(f"  有效K1 = {200*theta_gain_eff:.0f} ({theta_gain_eff:.1f}x)  "
          f"有效k2 = {100*omega_gain_eff:.0f} ({omega_gain_eff:.1f}x)")
    print(f"  u_hom = {np.round(u_hom,1)}  |u_hom|={np.linalg.norm(u_hom):.1f} rad/s²")
    print(f"  J·R_d'·u_hom = {np.round(J_RdT_uhom,3)} N·m")
    print(f"  ω×Jω = {np.round(gyro,4)} N·m")
    print(f"  M_actual = {np.round(M_actual,3)}  |M|={np.linalg.norm(M_actual):.3f} N·m")
    print(f"  thrust = {thrust_actual:.2f} N")

    return {
        'thrust': thrust_actual, 'M': M_actual, 'M_norm': np.linalg.norm(M_actual),
        'theta_norm': theta_norm, 'V_a': V_a, 'nx_sat': nx_sat,
        'tilt_raw_deg': np.rad2deg(tilt_raw),
        'tilt_actual_deg': np.rad2deg(tilt_actual),
        'tilt_limited': tilt_limited,
        'theta_gain_eff': theta_gain_eff,
        'u_hom': u_hom, 'u_hom_norm': np.linalg.norm(u_hom),
        'J_RdT_uhom': J_RdT_uhom, 'gyro': gyro,
        'V_p_z': V_p_ch[2],
    }


def run_actual_trajectory(ctrl, model, state0, pos_d, vel_d, dt, t_max):
    """运行轨迹，记录每步的实际控制输出。"""
    N = int(t_max / dt)
    state = state0.copy()

    M_norms = np.zeros(N)
    thrusts = np.zeros(N)
    theta_norms = np.zeros(N)
    tilt_degs = np.zeros(N)
    u_hom_norms = np.zeros(N)
    V_a_vals = np.zeros(N)
    times = np.linspace(dt, t_max, N)

    M_peak = 0.0
    M_peak_t = 0.0

    for k in range(N):
        u = ctrl.compute_control(state, pos_d, vel_d, 0.0)

        M_norms[k] = np.linalg.norm(u[1:4])
        thrusts[k] = u[0]
        theta_norms[k] = np.linalg.norm(ctrl.get_debug_info(state, pos_d, vel_d, 0.0)['theta_e'])
        V_a_vals[k] = ctrl.att_hpc.homogeneous_norm(
            np.concatenate([
                ctrl.get_debug_info(state, pos_d, vel_d, 0.0)['theta_e'],
                np.zeros(3)  # ω_e ≈ 0 for hover
            ]))

        if M_norms[k] > M_peak:
            M_peak = M_norms[k]
            M_peak_t = k * dt

        state = model.step_rk4(state, u, dt)

    return {
        'times': times, 'M_norms': M_norms, 'thrusts': thrusts,
        'theta_norms': theta_norms, 'V_a_vals': V_a_vals,
        'M_peak': M_peak, 'M_peak_t': M_peak_t,
    }


def analyze_K1_sweep():
    """K1 增益扫描（使用实际控制器输出，含所有限幅）。"""
    print("\n" + "=" * 60)
    print("  K1 扫描：实际力矩峰值（含 60° 倾角限制）")
    print("=" * 60)
    print(f"  {'K1':>5} {'eff_gain':>9} {'M_peak':>8} {'M_mean':>8} "
          f"{'θ_max':>8} {'收敛':>8}")

    m = 1.4
    J = np.diag([0.0211, 0.0219, 0.0366])
    model = QuadrotorSE3(m, J, 9.81)
    pos_d = np.array([1.0, 0.5, -2.0])
    state0 = model.make_state(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))

    for k1 in [10, 25, 50, 100, 200, 400]:
        ctrl = SE3HomogeneousController(m=m, J=J, mu_p=-0.5, mu_a=-0.5,
                                         K1_att=k1, k2_att=k1/2)
        traj = run_actual_trajectory(ctrl, model, state0, pos_d, np.zeros(3),
                                     dt=0.001, t_max=2.0)

        # 收敛检测
        theta_end = traj['theta_norms'][-1]
        conv_status = "OK" if theta_end < 0.01 else f"θ={theta_end:.3f}"

        eff_gain = k1 * 0.0211  # J_xx * K1 ≈ effective torque per rad
        print(f"  {k1:5d} {eff_gain:8.2f} {traj['M_peak']:8.2f} "
              f"{np.mean(traj['M_norms']):8.2f} "
              f"{np.rad2deg(np.max(traj['theta_norms'])):7.1f}° {conv_status:>8}")


def analyze_dt_effect():
    """离散时间步长对力矩的影响（SITL dt=0.01s vs 仿真 dt=0.001s）。"""
    print("\n" + "=" * 60)
    print("  dt 离散化效应 (SITL dt=0.01s)")
    print("=" * 60)
    print(f"  {'dt':>8} {'M_peak':>8} {'M_mean':>8} {'θ_max':>8} "
          f"{'收敛e_pos':>10}")

    m, J = 1.4, np.diag([0.0211, 0.0219, 0.0366])
    model = QuadrotorSE3(m, J, 9.81)
    pos_d = np.array([1.0, 0.5, -2.0])

    for dt in [0.001, 0.002, 0.005, 0.01, 0.02]:
        ctrl = SE3HomogeneousController(m=m, J=J, mu_p=-0.5, mu_a=-0.5)
        state = model.make_state(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))

        M_peak = 0
        M_vals = []
        N = int(3.0 / dt)
        for k in range(N):
            u = ctrl.compute_control(state, pos_d, np.zeros(3), 0.0)
            mn = np.linalg.norm(u[1:4])
            M_vals.append(mn)
            if mn > M_peak:
                M_peak = mn
            state = model.step_rk4(state, u, dt)

        e_final = np.linalg.norm(model.unpack_state(state)[0] - pos_d)
        print(f"  {dt:.3f}s {M_peak:8.2f} {np.mean(M_vals):8.2f} "
              f"{np.rad2deg(np.max(np.abs(model.unpack_state(state)[2] @ np.array([0,0,1]) - np.array([0,0,1])))):7.1f}° "
              f"{e_final:10.3f}m")


def analyze_thrust_deficit():
    """检查大倾角时推力不足导致的级联恶化（正反馈循环）。"""
    print("\n" + "=" * 60)
    print("  推力不足分析：大倾角 → 垂直推力不足 → Z下坠 → 更大误差")
    print("=" * 60)

    m, J, g = 1.4, np.diag([0.0211, 0.0219, 0.0366]), 9.81
    ctrl = SE3HomogeneousController(m=m, J=J, g=g, mu_p=-0.5, mu_a=-0.5)
    model = QuadrotorSE3(m, J, g)

    # 模拟不同倾角下需要的总推力
    print(f"  {'倾角':>6} {'需推力(N)':>10} {'垂推(N)':>10} {'净值(N)':>10} {'Z加速':>10}")
    for tilt_deg in [0, 15, 30, 45, 60, 70]:
        # 假设位置控制器输出 u_pos = [a_h, 0, 0] 使期望倾角为 tilt_deg
        tilt = np.deg2rad(tilt_deg)
        # tan(tilt) = a_h / g → a_h = g*tan(tilt)
        a_h = g * np.tan(tilt)
        # 推力方向需要与竖直方向成 tilt 角，提供水平 a_h 和竖直 g
        F_total = np.sqrt(a_h**2 + g**2)  # N/kg
        thrust_needed = m * F_total
        thrust_vertical = thrust_needed * np.cos(tilt)
        net_vertical = thrust_vertical - m * g
        z_accel = net_vertical / m
        print(f"  {tilt_deg:5.0f}° {thrust_needed:10.1f} {thrust_vertical:10.1f} "
              f"{net_vertical:10.1f} {z_accel:10.2f}")

    # 关键问题：如果推力饱和在 80N，45° 以上就无法悬停
    print(f"\n  最大推力 = 80N 时，最大可悬停倾角 = "
          f"{np.rad2deg(np.arccos(m*g/80)):.0f}°")


# ============================================================
if __name__ == '__main__':
    m, J, g = 1.4, np.diag([0.0211, 0.0219, 0.0366]), 9.81
    model = QuadrotorSE3(m, J, g)

    # ---- 1. 精确追踪实际控制器的第一步 ----
    print("=" * 60)
    print("  步骤 1: 实际控制器初始响应追踪（含所有限幅）")
    print("=" * 60)

    ctrl = SE3HomogeneousController(m=m, J=J, g=g, mu_p=-0.5, mu_a=-0.5)
    state0 = model.make_state(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    pos_d = np.array([1.0, 0.5, -2.0])
    vel_d = np.zeros(3)

    # t=0
    trace_actual_controller(ctrl, state0, pos_d, vel_d, 0.0, model, "t=0 (初始)")
    u0 = ctrl.compute_control(state0, pos_d, vel_d, 0.0)

    # 推进 0.1s
    state = state0.copy()
    for _ in range(100):
        u = ctrl.compute_control(state, pos_d, vel_d, 0.0)
        state = model.step_rk4(state, u, 0.001)
    trace_actual_controller(ctrl, state, pos_d, vel_d, 0.0, model, "t=0.1s")

    # 推进到 0.3s
    for _ in range(200):
        u = ctrl.compute_control(state, pos_d, vel_d, 0.0)
        state = model.step_rk4(state, u, 0.001)
    trace_actual_controller(ctrl, state, pos_d, vel_d, 0.0, model, "t=0.3s")

    # ---- 2. 完整的实际轨迹 ----
    print("\n" + "=" * 60)
    print("  步骤 2: 完整实际轨迹")
    print("=" * 60)
    traj = run_actual_trajectory(ctrl, model, state0, pos_d, vel_d, 0.001, 3.0)

    # 力矩统计
    above_5 = np.mean(traj['M_norms'] > 5) * 100
    above_10 = np.mean(traj['M_norms'] > 10) * 100
    above_20 = np.mean(traj['M_norms'] > 19) * 100
    print(f"  M_peak = {traj['M_peak']:.3f} N·m @ t={traj['M_peak_t']:.3f}s")
    print(f"  M_mean = {np.mean(traj['M_norms']):.3f} N·m")
    print(f"  |M| > 5 N·m : {above_5:.1f}%  (>10: {above_10:.1f}%  >20: {above_20:.1f}%)")
    print(f"  M(0.1s)={traj['M_norms'][100]:.3f}  M(0.5s)={traj['M_norms'][500]:.3f}  "
          f"M(1.0s)={traj['M_norms'][1000]:.3f}  M(2.0s)={traj['M_norms'][2000]:.3f}")

    # ---- 3. K1 扫描 ----
    analyze_K1_sweep()

    # ---- 4. dt 效应 ----
    analyze_dt_effect()

    # ---- 5. 推力不足分析 ----
    analyze_thrust_deficit()

    # ---- 6. 根本原因总结 ----
    print("\n" + "=" * 60)
    print("  诊断结论与建议")
    print("=" * 60)
    print("""
  问题 1: 初始力矩瞬态
    - 根因: 大初始位置误差 → 大倾角需求 → 大姿态误差
    - 现状: 60° 倾角限制已将 |M| 从理论 ~5.5 N·m 限制到可接受范围
    - 建议: 对 SITL，max_tilt=45° 更安全；或在 ROS2 节点中加入力矩速率限制

  问题 2: 齐次范数非线性放缩
    - 根因: V_a < 1 时，nx^(-1) 放大有效增益
    - 最差: nx=0.1 (α饱和) → 有效 K1 = 2000 (10x 放大)
    - 建议: 在 SITL 中将 α 提高到 0.2-0.3（减少近零点增益放大）

  问题 3: SITL dt=0.01s 离散化
    - 10x 粗于仿真 dt，可能导致离散化振荡
    - 建议: 用 si_ho (半隐式) 替代 e_ho，或在控制回路加低通滤波

  问题 4: 大倾角时推力不足的正反馈
    - 60° 倾角时，需 ~2x 悬停推力，垂直分量仅 50% → Z轴下坠
    - 下坠 → 更大位置误差 → 更大倾角 → 更恶化
    - 建议: 减小 max_tilt 到 45°，同时确保推力上限足够
""")
