#!/usr/bin/env python3
"""Plot UAV 3-loop homogeneous vs linear comparison from C++ CSV output."""

import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# Font setup
_cjk = [f.name for f in fm.fontManager.ttflist
        if any(k in f.name for k in ['SimHei','WenQuanYi','Noto Sans CJK',
                                      'Source Han','CJK','Heiti','Songti'])]
USE_EN = not bool(_cjk)
if _cjk: plt.rcParams['font.sans-serif'] = _cjk[:1] + ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def L(cn, en): return en if USE_EN else cn

# Find CSV files (default: cpp/build/)
build_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'build')

def load(fname):
    path = os.path.join(build_dir, fname)
    if not os.path.exists(path): path = fname
    return np.loadtxt(path, delimiter=',', skiprows=1)

print(L("读取 CSV 数据...", "Loading CSV data..."))
dz = load('uav_z_comparison_cpp.csv')
dy = load('uav_yaw_comparison_cpp.csv')
dxy = load('uav_xy_comparison_cpp.csv')

fig = plt.figure(figsize=(22, 15))

# === Z axis ===
t_z = dz[:,0]
ax = fig.add_subplot(3, 3, 1)
ax.plot(t_z, dz[:,3], 'r--', lw=1.5, label=L('线性 ez','Linear ez'))
ax.plot(t_z, dz[:,1], 'r', lw=2, label=L('HPIC ez','HPIC ez'))
ax.plot(t_z, dz[:,4], 'b--', lw=1.5, label=L('线性 vz','Linear vz'))
ax.plot(t_z, dz[:,2], 'b', lw=2, label=L('HPIC vz','HPIC vz'))
ax.axhline(0, color='k', ls='--', alpha=0.2)
ax.set_xlabel('t [s]'); ax.set_ylabel(L('状态','State'))
ax.set_title(L('Z轴 状态 (HPIC vs 线性PI)', 'Z States (HPIC vs Linear PI)'))
ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 2)
ez_hpic = np.sqrt(dz[:,1]**2 + dz[:,2]**2)
ez_lin  = np.sqrt(dz[:,3]**2 + dz[:,4]**2)
ax.semilogy(t_z, ez_lin, 'r--', lw=1.5, label=L('线性PI','Linear PI'))
ax.semilogy(t_z, ez_hpic, 'r', lw=2, label='HPIC')
ax.set_xlabel('t [s]'); ax.set_ylabel('||x||')
ax.set_title(L('Z轴 误差范数', 'Z Error Norm'))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 3)
ax.text(0.5, 0.5, L(f'HPIC:  ||x(T)||={ez_hpic[-1]:.4f}\n线性PI: ||x(T)||={ez_lin[-1]:.4f}\n提升: {ez_lin[-1]/ez_hpic[-1]:.1f}x',
                     f'HPIC:  ||x(T)||={ez_hpic[-1]:.4f}\nLinear: ||x(T)||={ez_lin[-1]:.4f}\nImprovement: {ez_lin[-1]/ez_hpic[-1]:.1f}x'),
        ha='center', va='center', fontsize=16, transform=ax.transAxes)
ax.set_title(L('Z轴 结果汇总', 'Z-axis Summary'))
ax.axis('off')

# === Yaw axis ===
t_y = dy[:,0]
ax = fig.add_subplot(3, 3, 4)
ax.plot(t_y, dy[:,3], 'r--', lw=1.5, label=L('线性 e_psi','Linear'))
ax.plot(t_y, dy[:,1], 'r', lw=2, label=L('HPC e_psi','HPC'))
ax.plot(t_y, dy[:,4], 'b--', lw=1.5, label=L('线性 wz','Linear wz'))
ax.plot(t_y, dy[:,2], 'b', lw=2, label=L('HPC wz','HPC wz'))
ax.axhline(0, color='k', ls='--', alpha=0.2)
ax.set_xlabel('t [s]'); ax.set_ylabel(L('状态','State'))
ax.set_title(L('Yaw轴 状态 (HPC vs 线性PD)', 'Yaw States (HPC vs Linear PD)'))
ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 5)
ey_hpc = np.sqrt(dy[:,1]**2 + dy[:,2]**2)
ey_lin = np.sqrt(dy[:,3]**2 + dy[:,4]**2)
ax.semilogy(t_y, ey_lin, 'r--', lw=1.5, label=L('线性PD','Linear PD'))
ax.semilogy(t_y, ey_hpc, 'r', lw=2, label='HPC')
ax.set_xlabel('t [s]'); ax.set_ylabel('||x||')
ax.set_title(L('Yaw轴 误差范数', 'Yaw Error Norm'))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 6)
ax.text(0.5, 0.5, L(f'HPC:  ||x(T)||={ey_hpc[-1]:.6f}\n线性PD: ||x(T)||={ey_lin[-1]:.6f}',
                     f'HPC:  ||x(T)||={ey_hpc[-1]:.6f}\nLinear: ||x(T)||={ey_lin[-1]:.6f}'),
        ha='center', va='center', fontsize=16, transform=ax.transAxes)
ax.set_title(L('Yaw轴 结果汇总', 'Yaw-axis Summary'))
ax.axis('off')

# === XY axis ===
t_xy = dxy[:,0]
ax = fig.add_subplot(3, 3, 7)
ax.plot(t_xy, dxy[:,9], 'r--', lw=1.5, label=L('LQR ex','LQR ex'))
ax.plot(t_xy, dxy[:,1], 'r', lw=2, label=L('HPC ex','HPC ex'))
ax.plot(t_xy, dxy[:,10], 'b--', lw=1.5, label=L('LQR ey','LQR ey'))
ax.plot(t_xy, dxy[:,2], 'b', lw=2, label=L('HPC ey','HPC ey'))
ax.axhline(0, color='k', ls='--', alpha=0.2)
ax.set_xlabel('t [s]'); ax.set_ylabel(L('位置误差 [m]','Position error [m]'))
ax.set_title(L('XY轴 位置误差 (HPC vs LQR)', 'XY Position Error (HPC vs LQR)'))
ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 8)
exy_hpc = np.sqrt(np.sum(dxy[:,1:9]**2, axis=1))
exy_lin = np.sqrt(np.sum(dxy[:,9:17]**2, axis=1))
ax.semilogy(t_xy, exy_lin, 'r--', lw=1.5, label=L('线性LQR','Linear LQR'))
ax.semilogy(t_xy, exy_hpc, 'r', lw=2, label='HPC')
ax.set_xlabel('t [s]'); ax.set_ylabel('||x||')
ax.set_title(L('XY轴 误差范数 (对数坐标)', 'XY Error Norm (log scale)'))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 9)
improvement = exy_lin[-1] / max(exy_hpc[-1], 1e-16)
ax.text(0.5, 0.5, L(f'HPC: ||x(T)||={exy_hpc[-1]:.2e}\nLQR:  ||x(T)||={exy_lin[-1]:.6f}\n'
                     f'HPC 有限时间收敛至机器零' if exy_hpc[-1] < 1e-10 else f'提升: {improvement:.1f}x',
                     f'HPC: ||x(T)||={exy_hpc[-1]:.2e}\nLQR:  ||x(T)||={exy_lin[-1]:.6f}\n'
                     f'HPC finite-time convergence' if exy_hpc[-1] < 1e-10 else f'Improvement: {improvement:.1f}x'),
        ha='center', va='center', fontsize=16, transform=ax.transAxes)
ax.set_title(L('XY轴 结果汇总', 'XY-axis Summary'))
ax.axis('off')

fig.suptitle(L('四旋翼三回路齐次控制器 — C++ 仿真 (虚线=线性, 实线=齐次)',
               'Quadrotor 3-Loop HPC — C++ Simulation (dashed=linear, solid=HPC)'),
             fontsize=15, fontweight='bold')
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), 'uav_comparison_cpp.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(L(f"图片已保存: {out}", f"Figure saved: {out}"))
try: plt.show()
except: pass
