"""
可视化工具 —— 四旋翼仿真结果的多控制器对比绘图
================================================

功能:
    - 3D 轨迹图（含 2D 回退方案）
    - 位置/速度/姿态误差的时间历程（线性坐标 + 对数坐标）
    - 控制输入（推力、力矩）对比图
    - 多控制器叠加对比模式

使用示例:
    plotter = ResultPlotter(save_dir='./figures')
    plotter.plot_comparison(results, title='Step Response Comparison')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # 非交互式后端（无需 GUI）
import matplotlib.pyplot as plt
import os


# ============================================================
# 统一配色方案（保证不同图中同一控制器的颜色一致）
# ============================================================
COLORS = {
    'lee_pd':     '#1f77b4',  # 蓝色 — Lee 几何 PD
    'hpc_mu0':    '#ff7f0e',  # 橙色 — HPC μ=0
    'hpc_mu_neg': '#2ca02c',  # 绿色 — HPC μ<0
    'linear':     '#d62728',  # 红色 — 线性控制器
    'hpc':        '#9467bd',  # 紫色 — HPC 通用
    'reference':  '#000000',  # 黑色 — 参考轨迹
}


class ResultPlotter:
    """
    仿真结果的高级可视化类。

    支持:
        - 多控制器对比面板（3 行 × 4 列）
        - 3D 轨迹图
        - 控制输入对比图
    """

    def __init__(self, save_dir=None, use_cjk=False):
        """
        参数:
            save_dir: 图片保存目录（None 则只显示不保存）
            use_cjk:  是否使用中文标签（需要系统安装中文字体）
        """
        self.save_dir = save_dir
        self.use_cjk = use_cjk

    def L(self, cn, en):
        """根据字体可用性返回中文或英文标签。"""
        return cn if self.use_cjk else en

    def plot_comparison(self, results, title='', filename='comparison.png',
                        figsize=(20, 15)):
        """
        完整的多控制器对比图（3 行 × 4 列面板）。

        面板布局:
            第 1 行: 位置误差 (x, y, z) + 位置误差范数（对数坐标）
            第 2 行: 速度误差 (x, y, z) + 速度误差范数（对数坐标）
            第 3 行: 姿态误差 (x, y, z) + 姿态误差范数（对数坐标）

        参数:
            results: dict, {控制器名称: {'t': t, 'state': state_log, 'input': u_log, 'ref': ref_log}}
            title:   总标题
            filename: 保存的文件名（相对于 save_dir）
            figsize:  图形尺寸
        """
        n_results = len(results)
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        labels = list(results.keys())

        # 为每个控制器分配颜色
        colors_map = {}
        for i, label in enumerate(labels):
            colors_map[label] = list(COLORS.values())[i % len(COLORS)]

        # ====== 第 1 行：位置误差 (x, y, z) + 位置范数 ======
        for idx, axis in enumerate(['x', 'y', 'z']):
            ax = axes[0, idx]
            for label in labels:
                r = results[label]
                pos = r['state'][:, idx]
                if r['ref'] is not None:
                    # 轨迹跟踪：误差 = 实际位置 - 参考位置
                    pos_ref = r['ref'][:, idx]
                    err = pos - pos_ref
                else:
                    # 阶跃/悬停：误差 = 实际位置（因为目标是零）
                    err = pos
                ax.plot(r['t'], err, color=colors_map[label], lw=1.5, label=label)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(f'e_{axis} [m]')
            ax.set_title(f'Position Error {axis.upper()}')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)

        # 位置误差范数（对数坐标 — 便于观察收敛速率）
        ax = axes[0, 3]
        for label in labels:
            r = results[label]
            pos = r['state'][:, 0:3]
            if r['ref'] is not None:
                pos_ref = r['ref'][:, 0:3]
                err = np.linalg.norm(pos - pos_ref, axis=1)
            else:
                err = np.linalg.norm(pos, axis=1)
            # 加微小偏移防止 log(0)
            ax.semilogy(r['t'], err + 1e-16, color=colors_map[label], lw=1.5, label=label)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('||e_pos|| [m]')
        ax.set_title('Position Error Norm (log)')
        ax.grid(True, alpha=0.3)

        # ====== 第 2 行：速度误差 ======
        for idx, axis in enumerate(['x', 'y', 'z']):
            ax = axes[1, idx]
            for label in labels:
                r = results[label]
                vel = r['state'][:, 3 + idx]
                if r['ref'] is not None:
                    vel_ref = r['ref'][:, 3 + idx]
                    err = vel - vel_ref
                else:
                    err = vel
                ax.plot(r['t'], err, color=colors_map[label], lw=1.5, label=label)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(f'e_v{axis} [m/s]')
            ax.set_title(f'Velocity Error {axis.upper()}')
            ax.grid(True, alpha=0.3)

        # 速度范数（对数坐标）
        ax = axes[1, 3]
        for label in labels:
            r = results[label]
            vel = r['state'][:, 3:6]
            if r['ref'] is not None:
                vel_ref = r['ref'][:, 3:6]
                err = np.linalg.norm(vel - vel_ref, axis=1)
            else:
                err = np.linalg.norm(vel, axis=1)
            ax.semilogy(r['t'], err + 1e-16, color=colors_map[label], lw=1.5)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('||e_vel|| [m/s]')
        ax.set_title('Velocity Error Norm (log)')
        ax.grid(True, alpha=0.3)

        # ====== 第 3 行：姿态误差（指数坐标 θ_e） ======
        from models.quadrotor_se3 import log_so3
        for idx, axis in enumerate(['x', 'y', 'z']):
            ax = axes[2, idx]
            for label in labels:
                r = results[label]
                state = r['state']
                attitude_err = np.zeros((state.shape[0], 3))
                for k in range(state.shape[0]):
                    R = state[k, 6:15].reshape(3, 3)
                    if r['ref'] is not None:
                        R_d = r['ref'][k, 6:15].reshape(3, 3)
                        R_e = R @ R_d.T    # 姿态误差旋转矩阵
                    else:
                        R_e = R             # 无参考时用当前姿态
                    attitude_err[k] = log_so3(R_e)  # 指数坐标
                ax.plot(r['t'], attitude_err[:, idx], color=colors_map[label],
                        lw=1.5, label=label)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(f'theta_e,{axis} [rad]')
            ax.set_title(f'Attitude Error {axis.upper()}')
            ax.grid(True, alpha=0.3)

        # 姿态范数（对数坐标）
        ax = axes[2, 3]
        for label in labels:
            r = results[label]
            state = r['state']
            att_norm = np.zeros(state.shape[0])
            for k in range(state.shape[0]):
                R = state[k, 6:15].reshape(3, 3)
                if r['ref'] is not None:
                    R_d = r['ref'][k, 6:15].reshape(3, 3)
                    R_e = R @ R_d.T
                else:
                    R_e = R
                att_norm[k] = np.linalg.norm(log_so3(R_e))
            ax.semilogy(r['t'], att_norm + 1e-16, color=colors_map[label], lw=1.5)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('|theta_e| [rad]')
        ax.set_title('Attitude Error Norm (log)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, filename), dpi=150,
                        bbox_inches='tight')
            print(f"Figure saved: {os.path.join(self.save_dir, filename)}")

        return fig, axes

    def plot_trajectory_3d(self, results, title='', filename='trajectory_3d.png'):
        """
        3D 轨迹对比图。

        含 matplotlib 3D 投影不可用时的 2D 回退方案（Bug #6 的绕过）。
        """
        fig = plt.figure(figsize=(12, 10))

        # 尝试 3D 投影
        try:
            ax = fig.add_subplot(111, projection='3d')
        except Exception:
            ax = None

        if ax is not None:
            # ---- 3D 模式 ----
            labels = list(results.keys())
            for i, label in enumerate(labels):
                r = results[label]
                pos = r['state'][:, 0:3]
                color = list(COLORS.values())[i % len(COLORS)]
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                        color=color, lw=1.5, label=label)

            # 参考轨迹（如果有）
            if results[labels[0]]['ref'] is not None:
                ref_pos = results[labels[0]]['ref'][:, 0:3]
                ax.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
                        color=COLORS['reference'], ls='--', lw=1.5, label='Reference')

            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.set_title(title)
            ax.legend(fontsize=9)
        else:
            # ---- 2D 回退方案（当 matplotlib 3D 不可用时）----
            # 绘制 XY 和 XZ 两个投影
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            labels = list(results.keys())
            for i, label in enumerate(labels):
                r = results[label]
                pos = r['state'][:, 0:3]
                color = list(COLORS.values())[i % len(COLORS)]
                ax1.plot(pos[:, 0], pos[:, 1], color=color, lw=1.5, label=label)
                ax2.plot(pos[:, 0], pos[:, 2], color=color, lw=1.5, label=label)
            ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]')
            ax2.set_xlabel('X [m]'); ax2.set_ylabel('Z [m]')
            ax1.set_title('XY Projection'); ax2.set_title('XZ Projection')
            ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, filename), dpi=150,
                        bbox_inches='tight')

        return fig, ax if ax is not None else (ax1, ax2)

    def plot_control_inputs(self, results, title='', filename='control_inputs.png',
                            figsize=(16, 10)):
        """
        控制输入对比图。

        2×2 子图: [推力, τ_x, τ_y, τ_z]
        """
        labels = list(results.keys())

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        input_names = ['Thrust [N]', 'τ_x [N·m]', 'τ_y [N·m]', 'τ_z [N·m]']

        for i, ax in enumerate(axes.flat):
            for j, label in enumerate(labels):
                r = results[label]
                u = r['input'][:, i]
                color = list(COLORS.values())[j % len(COLORS)]
                ax.plot(r['t'], u, color=color, lw=1.5, label=label)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(input_names[i])
            ax.set_title(input_names[i])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)

        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, filename), dpi=150,
                        bbox_inches='tight')

        return fig, axes
