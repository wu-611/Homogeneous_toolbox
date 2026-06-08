#!/usr/bin/env python3
"""
Phase 1 仿真脚本 —— SE(3) 全状态反馈齐次控制器对比
====================================================

功能:
    1. 阶跃悬停: 从原点飞到目标位置并悬停
    2. 螺旋轨迹跟踪: 跟踪上升螺旋线

对比控制器:
    - Lee PD (Lee et al. 2010): 几何 PD 基线
    - HPC μ=0: SE(3) 齐次控制器，指数收敛（退化为几何 PD 的齐次形式）
    - HPC μ<0: SE(3) 齐次控制器，有限时间收敛

输出:
    - figures/step_hover_comparison.png      : 阶跃响应 12 面板对比图
    - figures/spiral_trajectory_comparison.png : 螺旋跟踪对比图
    - figures/spiral_trajectory_3d.png        : 3D 轨迹图

使用方法:
    python3 scripts/run_phase1_simulations.py

仿真参数:
    - 积分方法: RK4, dt = 0.001s (1kHz)
    - 四旋翼: m=1.4kg, J=diag([0.0211, 0.0219, 0.0366])
    - 阶跃目标: [1.0, 0.5, -2.0] m, 仿真 8s
    - 螺旋轨迹: r=2m, ω=0.5rad/s, Z 上升 0.1m/s, 仿真 10s
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.quadrotor_se3 import QuadrotorSE3, exp_so3, log_so3
from simulation.simulator import Simulator
from controllers.se3_homogeneous_full import SE3HomogeneousController, LeeGeometricPD
from visualization.plotter import ResultPlotter


# ============================================================
# 场景 1：阶跃悬停响应
# ============================================================
def run_step_response(model, controller, pos_d, yaw_d=0.0, T=8.0,
                      label='controller'):
    """
    运行阶跃悬停仿真。

    从原点 [0,0,0] 出发，目标位置 pos_d，目标速度 0。
    测试控制器的瞬态响应和稳态精度。
    """
    sim = Simulator(model, dt=0.001, t_max=T)

    # 初始状态：原点悬停
    pos0 = np.zeros(3)
    vel0 = np.zeros(3)
    R0 = np.eye(3)
    omega0 = np.zeros(3)
    state0 = model.make_state(pos0, vel0, R0, omega0)
    vel_d = np.zeros(3)

    # 控制器闭包：*args 吸收 Simulator 可能传入的参考轨迹参数
    def ctrl(t, state, *args):
        return controller.compute_control(state, pos_d, vel_d, yaw_d)

    result = sim.run(ctrl, state0)
    return result


# ============================================================
# 场景 2：螺旋轨迹跟踪
# ============================================================
def run_trajectory_tracking(model, controller, T=10.0, label='controller'):
    """
    运行螺旋轨迹跟踪仿真。

    轨迹: r=2m, ω=0.5rad/s 的水平圆 + Z 轴缓慢上升 (-0.1 m/s)
    初始状态: 在轨迹起点 [2, 0, -2] 上，初速度匹配轨迹速度。
    """
    sim = Simulator(model, dt=0.001, t_max=T)

    # 初始状态设置为轨迹起点（减少初始瞬态）
    pos0 = np.array([2.0, 0.0, -2.0])    # 轨迹起点: [r, 0, z0]
    vel0 = np.array([0.0, 1.0, -0.1])    # 轨迹初速度: [0, r·ω, ż]
    R0 = np.eye(3)
    omega0 = np.zeros(3)
    state0 = model.make_state(pos0, vel0, R0, omega0)

    def ctrl(t, state, *args):
        # 螺旋轨迹参数方程:
        #   x(t) = r·cos(ω·t)
        #   y(t) = r·sin(ω·t)
        #   z(t) = z0 + ż·t
        r = 2.0
        omega_traj = 0.5                    # 轨迹角频率 [rad/s]
        pos_d = np.array([r * np.cos(omega_traj * t),
                          r * np.sin(omega_traj * t),
                          -2.0 - 0.1 * t])  # Z 缓慢上升
        vel_d = np.array([-r * omega_traj * np.sin(omega_traj * t),
                          r * omega_traj * np.cos(omega_traj * t),
                          -0.1])
        yaw_d = omega_traj * t              # 偏航对准前进方向
        return controller.compute_control(state, pos_d, vel_d, yaw_d)

    # 参考轨迹生成器（用于误差计算）
    def ref_traj(t):
        r = 2.0
        omega_traj = 0.5
        pos_d = np.array([r * np.cos(omega_traj * t),
                          r * np.sin(omega_traj * t),
                          -2.0 - 0.1 * t])
        vel_d = np.array([-r * omega_traj * np.sin(omega_traj * t),
                          r * omega_traj * np.cos(omega_traj * t),
                          -0.1])
        R_d = np.eye(3)                     # 简化：忽略期望姿态
        omega_d = np.array([0., 0., omega_traj])
        return pos_d, vel_d, R_d, omega_d

    result = sim.run(ctrl, state0, ref_traj=ref_traj)
    return result


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 65)
    print("Phase 1 仿真：SE(3) 几何齐次控制器 — 全状态反馈")
    print("=" * 65)

    # ---- 四旋翼物理参数 ----
    m = 1.4
    J = np.diag([0.0211, 0.0219, 0.0366])
    g = 9.81
    model = QuadrotorSE3(m, J, g)

    # ---- 创建三个对比控制器 ----
    lee_pd = LeeGeometricPD(m, J, g)                     # 基线: Lee 几何 PD
    hpc_mu0 = SE3HomogeneousController(m, J, g,          # HPC μ=0（指数收敛）
                                        mu_p=0.0, mu_a=0.0)
    hpc_mu_neg = SE3HomogeneousController(m, J, g,       # HPC μ<0（有限时间）
                                           mu_p=-0.5, mu_a=-0.5)

    controllers = {
        'Lee PD': lee_pd,
        'HPC mu=0': hpc_mu0,
        'HPC mu<0': hpc_mu_neg,
    }

    # ====== 场景 1：阶跃悬停 ======
    print("\n>>> 场景 1: 阶跃悬停 (目标位置 [1, 0.5, -2] m)")
    pos_d_step = np.array([1.0, 0.5, -2.0])

    results_step = {}
    for name, ctrl in controllers.items():
        print(f"  运行 {name}...")
        results_step[name] = run_step_response(model, ctrl, pos_d_step, T=8.0)

    # 打印终态位置误差
    print("\n  终态位置误差 ||e_pos||:")
    for name, r in results_step.items():
        pos_final = r['state'][-1, 0:3]
        err = np.linalg.norm(pos_final - pos_d_step)
        print(f"    {name:12s}: {err:.4f} m")

    # ====== 场景 2：螺旋轨迹跟踪 ======
    print("\n>>> 场景 2: 螺旋轨迹跟踪 (r=2m, ω=0.5rad/s)")
    results_spiral = {}
    for name, ctrl in controllers.items():
        print(f"  运行 {name}...")
        results_spiral[name] = run_trajectory_tracking(model, ctrl, T=10.0)

    # 打印 RMS 跟踪误差
    print("\n  RMS 位置跟踪误差:")
    for name, r in results_spiral.items():
        pos = r['state'][:, 0:3]
        ref = r['ref'][:, 0:3]
        rms_err = np.sqrt(np.mean(np.sum((pos - ref)**2, axis=1)))
        print(f"    {name:12s}: RMS = {rms_err:.4f} m")

    # ====== 生成对比图 ======
    print("\n>>> 生成对比图...")

    plotter = ResultPlotter(save_dir='./figures', use_cjk=False)

    # 阶跃响应对比
    plotter.plot_comparison(
        results_step,
        title='Step Hover: Lee PD vs HPC (mu=0) vs HPC (mu<0)',
        filename='step_hover_comparison.png'
    )

    # 螺旋轨迹对比
    plotter.plot_comparison(
        results_spiral,
        title='Spiral Trajectory: Lee PD vs HPC (mu=0) vs HPC (mu<0)',
        filename='spiral_trajectory_comparison.png'
    )

    # 3D 轨迹
    plotter.plot_trajectory_3d(
        results_spiral,
        title='Spiral Trajectory Tracking',
        filename='spiral_trajectory_3d.png'
    )

    # ====== 定量指标汇总 ======
    print("\n" + "=" * 65)
    print("定量指标汇总")
    print("=" * 65)

    for name in controllers:
        r_step = results_step[name]

        # 阶跃 ISE（积分平方误差）
        pos_step = r_step['state'][:, 0:3]
        pos_d_arr = np.tile(pos_d_step, (pos_step.shape[0], 1))
        ise_step = np.trapezoid(np.sum((pos_step - pos_d_arr)**2, axis=1),
                                r_step['t'])

        # 控制能量（平方根积分）
        u_step = r_step['input']
        energy_step = np.sqrt(np.trapezoid(np.sum(u_step**2, axis=1), r_step['t']))

        print(f"\n  {name}:")
        print(f"    阶跃 ISE (积分平方误差):    {ise_step:.4f} m²·s")
        print(f"    阶跃控制能量:               {energy_step:.2f}")

    print(f"\n图片已保存到 ./figures/")
    print("完成!")


if __name__ == '__main__':
    main()
