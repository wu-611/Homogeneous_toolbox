"""
数值验证脚本：从仿真轨迹估计 Lyapunov 衰减率 ρ̃ 并验证收敛时间上界。

策略：
    1. 纯 Z 轴悬停 (无姿态耦合) → 估计 ρ̃_p
    2. 纯姿态阶跃 (悬停位置，初始姿态误差) → 估计 ρ̃_a
    3. 组合阶跃 → 验证级联上界 T_total ≤ T_a + T_p

对应理论：docs/Phase2_有限时间收敛理论推导.md
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.quadrotor_se3 import QuadrotorSE3, log_so3, exp_so3, hat
from controllers.se3_homogeneous_full import SE3HomogeneousController
from design.attitude_command import compute_attitude_error, compute_omega_error


def compute_vp_single(x, hpc):
    """Compute V_p for a single position channel (e_p, e_v)."""
    from hcs_toolbox_py import hnorm
    return hnorm(x, hpc.Gd, hpc.P)


def compute_va(xi, hpc):
    """Compute V_a for attitude state xi = [theta_e; omega_e]."""
    from hcs_toolbox_py import hnorm
    return hnorm(xi, hpc.Gd, hpc.P)


def estimate_rho_robust(V_traj, dt, mu, frac=0.8):
    """
    从 V(t) 轨迹估计 ρ̃。

    使用所有 V > alpha_sat 的数据点（排除饱和区），因为这些点
    满足理想的齐次衰减模型 V̇ = -ρ̃ V^(1+μ)。

    方法: 拟合 V(t)^(-mu) = V(0)^(-mu) + ρ̃·(-mu)·t (线性)
    注意: V^(-mu) = V(t)^(0.5) 随时间线性递减（因为 μ=-0.5 < 0）
         斜率 = ρ̃·μ < 0, 所以 ρ̃ = slope/μ > 0
    """
    n = len(V_traj)

    # 使用 V > alpha_sat 的所有点（排除 alpha 饱和区，那里齐次模型不成立）
    alpha_sat = 0.15
    valid = V_traj > alpha_sat

    # 取前 frac 比例的 valid 点（避免尾部接近饱和区的非线性效应）
    valid_indices = np.where(valid)[0]
    if len(valid_indices) < 10:
        return 0.0, 0.0

    cut = int(len(valid_indices) * frac)
    use_indices = valid_indices[:cut]

    V_use = V_traj[use_indices]
    t_use = use_indices * dt

    # 变换：Y = V^(-mu)
    Y = V_use ** (-mu)
    A = np.column_stack([np.ones_like(t_use), t_use])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    intercept, slope = coeffs
    # Y = V^(-mu) = V0^(-mu) + ρ̃·μ·t
    # slope = ρ̃·μ  →  ρ̃ = slope/μ (μ<0, slope<0 → ρ̃>0)
    rho = slope / mu if abs(mu) > 1e-10 else 0.0

    Y_pred = intercept + slope * t_use
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    return max(rho, 1e-6), r_sq


def compute_t_bound(V0, rho, mu):
    """有限时间收敛上界: T ≤ V0^(-mu) / (-rho * mu)."""
    if mu >= 0 or rho <= 0 or V0 <= 0:
        return np.inf
    return V0 ** (-mu) / (-rho * mu)


def test_pure_z(mu_p=-0.5, dt=0.001, t_max=5.0):
    """
    纯 Z 轴悬停测试：初始 e_z = -1m, e_vz = 0.

    无姿态耦合 (推力始终竖直), 用于估计 ρ̃_p.
    """
    m = 1.4
    J = np.diag([0.0211, 0.0219, 0.0366])
    g = 9.81

    ctrl = SE3HomogeneousController(m=m, J=J, g=g, mu_p=mu_p, mu_a=0.0)
    model = QuadrotorSE3(m=m, J=J, g=g)

    # 初始：z=0, 目标：z=-2. 纯 Z 轴, R=I, omega=0.
    state = model.make_state(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    pos_d = np.array([0.0, 0.0, -2.0])
    vel_d = np.zeros(3)

    N = int(t_max / dt)
    V_z = np.zeros(N + 1)
    e_z_traj = np.zeros(N + 1)

    for k in range(N + 1):
        di = ctrl.get_debug_info(state, pos_d, vel_d, 0.0)
        e_z_traj[k] = di['e_pos'][2]
        V_z[k] = compute_vp_single(
            np.array([di['e_pos'][2], di['e_vel'][2]]), ctrl.pos_hpc)

        if k < N:
            u = ctrl.compute_control(state, pos_d, vel_d, 0.0)
            state = model.step_rk4(state, u, dt)

    rho_p, r_sq = estimate_rho_robust(V_z, dt, mu_p, frac=0.3)
    V0 = V_z[0]
    T_bound = compute_t_bound(V0, rho_p, mu_p)

    conv_idx = np.argmax(np.abs(e_z_traj) < 0.01)
    T_actual = conv_idx * dt if conv_idx > 0 else np.inf

    return {
        'rho_p': rho_p, 'r_sq': r_sq, 'V0': V0,
        'T_bound': T_bound, 'T_actual': T_actual,
        'V_traj': V_z, 'e_traj': e_z_traj
    }


def test_pure_attitude(mu_a=-0.5, dt=0.001, t_max=3.0):
    """
    纯姿态阶跃测试：悬停位置，初始姿态误差 30°.

    无位置耦合 (期望位置不变), 用于估计 ρ̃_a.
    """
    m = 1.4
    J = np.diag([0.0211, 0.0219, 0.0366])
    g = 9.81

    ctrl = SE3HomogeneousController(m=m, J=J, g=g, mu_p=0.0, mu_a=mu_a)
    model = QuadrotorSE3(m=m, J=J, g=g)

    # 初始姿态：绕 Y 轴旋转 30° = 0.524 rad
    theta0 = np.array([0.0, 0.524, 0.0])
    R0 = exp_so3(theta0)
    state = model.make_state(np.zeros(3), np.zeros(3), R0, np.zeros(3))

    pos_d = np.array([0.0, 0.0, -2.0])
    vel_d = np.zeros(3)

    N = int(t_max / dt)
    V_a = np.zeros(N + 1)
    theta_norms = np.zeros(N + 1)

    # R_d 由悬停推力方向确定
    F_des = np.array([0.0, 0.0, m * g])
    b3_des = F_des / np.linalg.norm(F_des)
    from design.attitude_command import compute_desired_attitude
    R_d = compute_desired_attitude(b3_des, 0.0)

    for k in range(N + 1):
        pos, vel, R, omega = model.unpack_state(state)
        theta_e = compute_attitude_error(R, R_d)
        omega_e = compute_omega_error(omega, np.zeros(3), R, R_d)
        xi = np.concatenate([theta_e, omega_e])
        V_a[k] = compute_va(xi, ctrl.att_hpc)
        theta_norms[k] = np.linalg.norm(theta_e)

        if k < N:
            u = ctrl.compute_control(state, pos_d, vel_d, 0.0)
            state = model.step_rk4(state, u, dt)

    rho_a, r_sq = estimate_rho_robust(V_a, dt, mu_a, frac=0.3)
    V0 = V_a[0]
    T_bound = compute_t_bound(V0, rho_a, mu_a)

    conv_idx = np.argmax(theta_norms < 0.005)  # 0.3 deg
    T_actual = conv_idx * dt if conv_idx > 0 else np.inf

    return {
        'rho_a': rho_a, 'r_sq': r_sq, 'V0': V0,
        'T_bound': T_bound, 'T_actual': T_actual,
        'V_traj': V_a, 'theta_norms': theta_norms
    }


def test_cascaded(mu_p=-0.5, mu_a=-0.5, dt=0.001, t_max=8.0):
    """
    级联阶跃测试：同时有位置和姿态误差 → 验证 T_total ≤ T_a + T_p.

    初始：pos=[0,0,0], 目标：[1, 0.5, -2].
    初始姿态 I (无初始姿态误差，但位置控制会产生姿态误差).
    """
    m = 1.4
    J = np.diag([0.0211, 0.0219, 0.0366])
    g = 9.81

    ctrl = SE3HomogeneousController(m=m, J=J, g=g, mu_p=mu_p, mu_a=mu_a)
    model = QuadrotorSE3(m=m, J=J, g=g)

    state = model.make_state(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    pos_d = np.array([1.0, 0.5, -2.0])
    vel_d = np.zeros(3)

    N = int(t_max / dt)
    V_p_traj = np.zeros(N + 1)
    V_a_traj = np.zeros(N + 1)
    e_pos_norms = np.zeros(N + 1)
    theta_norms = np.zeros(N + 1)

    for k in range(N + 1):
        di = ctrl.get_debug_info(state, pos_d, vel_d, 0.0)
        e_pos_norms[k] = np.linalg.norm(di['e_pos'])

        # 位置 Lyapunov (RMS of 3 channels)
        V_p_ch = np.zeros(3)
        for ch in range(3):
            V_p_ch[ch] = compute_vp_single(
                np.array([di['e_pos'][ch], di['e_vel'][ch]]), ctrl.pos_hpc)
        V_p_traj[k] = np.sqrt(np.sum(V_p_ch ** 2))

        # 姿态 Lyapunov
        xi = np.concatenate([di['theta_e'], di['omega_e']])
        V_a_traj[k] = compute_va(xi, ctrl.att_hpc)
        theta_norms[k] = np.linalg.norm(di['theta_e'])

        if k < N:
            u = ctrl.compute_control(state, pos_d, vel_d, 0.0)
            state = model.step_rk4(state, u, dt)

    # 实际收敛时间 (5% 初始误差)
    e0 = e_pos_norms[0]
    converged = e_pos_norms < max(0.01, 0.05 * e0)
    T_actual = np.argmax(converged) * dt if np.any(converged) else np.inf

    return {
        'V_p': V_p_traj, 'V_a': V_a_traj,
        'e_pos_norms': e_pos_norms, 'theta_norms': theta_norms,
        'T_pos_actual': T_actual
    }


def main():
    print("=" * 70)
    print("齐次级联控制 Lyapunov 衰减率估计与收敛时间验证")
    print("=" * 70)

    # ---- 测试 1: 纯 Z 轴 ----
    print("\n[1] 纯 Z 轴悬停 (估计 ρ̃_p)")
    print("-" * 40)
    r_z = test_pure_z(mu_p=-0.5)
    print(f"  V_p(0)  = {r_z['V0']:.3f}")
    print(f"  ρ̃_p    = {r_z['rho_p']:.3f}  (R² = {r_z['r_sq']:.4f})")
    print(f"  T_bound = {r_z['T_bound']:.3f}s")
    print(f"  T_actual = {r_z['T_actual']:.3f}s")
    print(f"  T_bound/T_actual = {r_z['T_bound']/r_z['T_actual']:.2f}x")

    # ---- 测试 2: 纯姿态 ----
    print("\n[2] 纯姿态阶跃 30° (估计 ρ̃_a)")
    print("-" * 40)
    r_a = test_pure_attitude(mu_a=-0.5)
    print(f"  V_a(0)  = {r_a['V0']:.3f}")
    print(f"  ρ̃_a    = {r_a['rho_a']:.3f}  (R² = {r_a['r_sq']:.4f})")
    print(f"  T_bound = {r_a['T_bound']:.3f}s")
    print(f"  T_actual = {r_a['T_actual']:.3f}s")
    print(f"  T_bound/T_actual = {r_a['T_bound']/r_a['T_actual']:.2f}x")

    # ---- 测试 3: 级联阶跃 ----
    print("\n[3] 级联阶跃响应 (验证 T_total)")
    print("-" * 40)
    r_c = test_cascaded(mu_p=-0.5, mu_a=-0.5)
    print(f"  T_pos_actual = {r_c['T_pos_actual']:.3f}s")
    T_total_bound = r_z['T_bound'] + r_a['T_bound']
    print(f"  T_total_bound = {T_total_bound:.3f}s (T_p + T_a)")

    if T_total_bound >= r_c['T_pos_actual']:
        print(f"  ✓ T_total_bound ({T_total_bound:.3f}s) ≥ T_actual "
              f"({r_c['T_pos_actual']:.3f}s)")

    # ---- 测试 4: 参数扫描 ----
    if '--scan' in sys.argv:
        print("\n" + "=" * 70)
        print("[4] 参数扫描: μ_p × μ_a")
        print("=" * 70)
        mu_vals = [-0.3, -0.5, -0.7]
        print(f"\n{'μ_p':>6} {'μ_a':>6} {'T_bound_p':>10} {'T_bound_a':>10} "
              f"{'T_bound_total':>14} {'T_actual':>10}")
        print("-" * 60)

        for mu_p in mu_vals:
            for mu_a in mu_vals:
                rz = test_pure_z(mu_p=mu_p, t_max=5.0)
                ra = test_pure_attitude(mu_a=mu_a, t_max=3.0)
                rc = test_cascaded(mu_p=mu_p, mu_a=mu_a, t_max=6.0)

                T_bound_total = rz['T_bound'] + ra['T_bound']
                valid = "OK" if T_bound_total >= rc['T_pos_actual'] else "FAIL"

                print(f"{mu_p:6.1f} {mu_a:6.1f} {rz['T_bound']:10.3f} "
                      f"{ra['T_bound']:10.3f} {T_bound_total:14.3f} "
                      f"{rc['T_pos_actual']:10.3f}  {valid}")

    # ---- 输出论文用参数 ----
    print("\n" + "=" * 70)
    print("论文用关键参数汇总")
    print("=" * 70)
    print(f"  ρ̃_p  = {r_z['rho_p']:.2f}  (R²={r_z['r_sq']:.3f})")
    print(f"  ρ̃_a  = {r_a['rho_a']:.2f}  (R²={r_a['r_sq']:.3f})")
    print(f"  T_p   ≤ {r_z['T_bound']:.2f} s  (μ_p = -0.5)")
    print(f"  T_a   ≤ {r_a['T_bound']:.3f} s  (μ_a = -0.5)")
    print(f"  T_total ≤ {r_z['T_bound'] + r_a['T_bound']:.2f} s")
    print(f"\n  验证: T_bound/T_actual (位置) = {r_z['T_bound']/r_z['T_actual']:.1f}x")
    print(f"         T_bound/T_actual (姿态) = {r_a['T_bound']/r_a['T_actual']:.1f}x")


if __name__ == '__main__':
    main()
