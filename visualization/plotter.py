"""
Visualization tools for quadrotor simulation results.

Supports:
- 3D trajectory plots
- Position/velocity error convergence plots (linear and log scale)
- Control input plots
- Multi-controller comparison overlay
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# Color palette for consistent multi-controller plots
COLORS = {
    'lee_pd':     '#1f77b4',  # blue
    'hpc_mu0':    '#ff7f0e',  # orange
    'hpc_mu_neg': '#2ca02c',  # green
    'linear':     '#d62728',  # red
    'hpc':        '#9467bd',  # purple
    'reference':  '#000000',  # black
}

LINESTYLES = {
    'linear':     '--',
    'hpc':        '-',
    'reference':  ':',
}


def plot_trajectory_3d(ax, pos_log, label=None, color=None, ls='-', lw=1.5,
                       alpha=1.0):
    """Plot 3D trajectory on existing axes."""
    ax.plot(pos_log[:, 0], pos_log[:, 1], pos_log[:, 2],
            color=color, ls=ls, lw=lw, label=label, alpha=alpha)


def plot_error_convergence(ax, t, error_log, label=None, color=None, ls='-',
                           lw=1.5, log_scale=True):
    """Plot error norm over time."""
    if log_scale:
        ax.semilogy(t, error_log, color=color, ls=ls, lw=lw, label=label)
    else:
        ax.plot(t, error_log, color=color, ls=ls, lw=lw, label=label)


def plot_control_input(ax, t, u_log, labels=None, colors=None):
    """Plot control inputs as stacked subplots."""
    n_inputs = u_log.shape[1]
    if labels is None:
        labels = [f'u_{i}' for i in range(n_inputs)]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_inputs))

    for i in range(n_inputs):
        ax.plot(t, u_log[:, i], color=colors[i], lw=1.5, label=labels[i])


class ResultPlotter:
    """
    High-level plotter for quadrotor simulation comparisons.

    Usage:
        plotter = ResultPlotter(save_dir='./figures')
        plotter.plot_comparison(results_dict, title='...')
    """

    def __init__(self, save_dir=None, use_cjk=False):
        self.save_dir = save_dir
        self.use_cjk = use_cjk

        if use_cjk:
            self._setup_cjk_fonts()

    def _setup_cjk_fonts(self):
        """Try to configure CJK fonts."""
        import matplotlib.font_manager as fm
        cjk = [f.name for f in fm.fontManager.ttflist
               if any(k in f.name for k in
                      ['SimHei', 'WenQuanYi', 'Noto Sans CJK',
                       'Source Han', 'CJK', 'Heiti', 'Songti'])]
        if cjk:
            plt.rcParams['font.sans-serif'] = cjk[:1] + ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def L(self, cn, en):
        """Return Chinese or English label based on font availability."""
        return en if not self.use_cjk else cn

    def plot_comparison(self, results, title='', filename='comparison.png',
                        figsize=(20, 15)):
        """
        Full comparison plot for multiple controllers.

        Parameters
        ----------
        results : dict
            {label: {'t': t, 'state': state_log, 'input': input_log, 'ref': ref_log}}
            Each state_log is (N+1, 18) with SE(3) state.
        title : str
            Overall figure title.
        filename : str
            Output filename (relative to save_dir).
        figsize : tuple
            Figure size.
        """
        n_results = len(results)
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        labels = list(results.keys())
        colors_map = {}
        for i, label in enumerate(labels):
            colors_map[label] = list(COLORS.values())[i % len(COLORS)]

        t_min = min(results[l]['t'][-1] for l in labels)

        # Row 1: Position errors (x, y, z)
        for idx, axis in enumerate(['x', 'y', 'z']):
            ax = axes[0, idx]
            for label in labels:
                r = results[label]
                pos = r['state'][:, idx]
                if r['ref'] is not None:
                    pos_ref = r['ref'][:, idx]
                    err = pos - pos_ref
                else:
                    err = pos
                ax.plot(r['t'], err, color=colors_map[label], lw=1.5, label=label)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(f'e_{axis} [m]')
            ax.set_title(f'Position Error {axis.upper()}')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)

        # Row 1, col 4: Position error norm (log scale)
        ax = axes[0, 3]
        for label in labels:
            r = results[label]
            pos = r['state'][:, 0:3]
            if r['ref'] is not None:
                pos_ref = r['ref'][:, 0:3]
                err = np.linalg.norm(pos - pos_ref, axis=1)
            else:
                err = np.linalg.norm(pos, axis=1)
            ax.semilogy(r['t'], err + 1e-16, color=colors_map[label], lw=1.5, label=label)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('||e_pos|| [m]')
        ax.set_title('Position Error Norm (log)')
        ax.grid(True, alpha=0.3)

        # Row 2: Velocity errors
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

        # Row 2, col 4: Velocity error norm
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

        # Row 3: Attitude error (extracted via log_so3 from R_e = R * R_d')
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
                        R_e = R @ R_d.T
                    else:
                        R_e = R
                    attitude_err[k] = log_so3(R_e)
                ax.plot(r['t'], attitude_err[:, idx], color=colors_map[label],
                        lw=1.5, label=label)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(f'theta_e,{axis} [rad]')
            ax.set_title(f'Attitude Error {axis.upper()}')
            ax.grid(True, alpha=0.3)

        # Row 3, col 4: Attitude error norm
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
            ax.semilogy(r['t'], att_norm + 1e-16, color=colors_map[label],
                        lw=1.5)
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
        """3D trajectory comparison plot."""
        fig = plt.figure(figsize=(12, 10))
        try:
            ax = fig.add_subplot(111, projection='3d')
        except Exception:
            # Fallback: 2D projections
            ax = None

        if ax is not None:
            labels = list(results.keys())
            for i, label in enumerate(labels):
                r = results[label]
                pos = r['state'][:, 0:3]
                color = list(COLORS.values())[i % len(COLORS)]
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                        color=color, lw=1.5, label=label)

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
            # Fallback: plot XY and XZ projections
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

        return fig, ax

    def plot_control_inputs(self, results, title='', filename='control_inputs.png',
                            figsize=(16, 10)):
        """Control input comparison across controllers."""
        labels = list(results.keys())
        n_controllers = len(labels)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        input_names = ['Thrust [N]', 'tau_x [N·m]', 'tau_y [N·m]', 'tau_z [N·m]']

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
