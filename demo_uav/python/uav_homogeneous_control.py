#!/usr/bin/env python3
"""
uav_homogeneous_control.py — 四旋翼无人机三回路齐次控制器仿真

复现 Compute_All_HPC_Params.m，包含：
  Z 回路 (2D): 高度+速度         → HPIC (齐次 PI)
  Yaw回路 (2D): 偏航角+角速度     → HPC  (齐次 P)
  XY 回路 (8D): 位置+速度+姿态+角速度 → HPC  (齐次 P, LQR)

每个回路同时仿真线性控制器与齐次控制器，对比阶跃响应。

运行:  python3 uav_homogeneous_control.py
"""

import sys, os
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ---- 导入齐次控制工具箱 ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hcs_toolbox_py import lpc2hpc, lpic2hpic, ZOH, hnorm, e_hpc, e_hpic

# ---- 字体：优先 CJK，否则英文 ----
_cjk = [f.name for f in fm.fontManager.ttflist
        if any(k in f.name for k in ['SimHei','WenQuanYi','Noto Sans CJK',
                                      'Source Han','CJK','Heiti','Songti'])]
USE_EN = not bool(_cjk)
if _cjk:
    plt.rcParams['font.sans-serif'] = _cjk[:1] + ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---- 标签辅助 ----
def L(cn, en):
    """根据字体可用性返回中文或英文标签"""
    return en if USE_EN else cn

# ===========================================================================
# 全局参数
# ===========================================================================
G  = 9.8
DT = 0.001

print("=" * 65)
print(L("四旋翼三回路齐次控制器 (HPC/HPIC) — 复现 Compute_All_HPC_Params.m",
        "Quadrotor 3-Loop Homogeneous Controller — reproducing Compute_All_HPC_Params.m"))
print("=" * 65)

# ===========================================================================
# 1. Z 轴 HPIC (高度) — 齐次比例-积分
# ===========================================================================
print(f"\n>>> 1. Z-axis HPIC (altitude)")

m   = 1.4
A_z = np.array([[0., 1.], [0., 0.]])
B_z = np.array([[0.], [1. / m]])
Kz_lin  = np.array([[-5., -2.]])
Kiz_lin = np.array([[-0.1, 0.]])

K0_z, G0_z, P_z, Ki_new, mu_min_z, mu_max_z = lpic2hpic(A_z, B_z, Kz_lin, Kiz_lin)
mu_z    = -0.5
Gd_z    = np.eye(2) + mu_z * G0_z
Kz_nl   = Kz_lin - K0_z

print(f"  mu={mu_z}, 容许范围 [{mu_min_z:.4f}, {mu_max_z:.4f}]")
print(f"  K0_z={K0_z[0]}, Ki_new={Ki_new[0]}")

# -- Z 仿真 (ZOH) --
Ah_z, Bh_z = ZOH(DT, A_z, B_z); Bh_z = Bh_z.flatten()
Tz, Nz = 5.0, int(5.0 / DT)
hn_z = lambda x: hnorm(x, Gd_z, P_z)

def sim_z(x0, use_hpic=True):
    x, vi = x0.copy(), 0.0
    X = np.zeros((2, Nz+1)); X[:,0] = x
    for k in range(Nz):
        if use_hpic:
            uh, ui = e_hpic(x, K0_z, Kz_nl, Ki_new, Gd_z, mu_z, hn_z, alpha=0.1)
            u = uh[0] + vi; vi += DT * ui[0]
        else:
            u = (Kz_lin @ x)[0] + vi; vi += DT * (Kiz_lin @ x)[0]
        x = Ah_z @ x + Bh_z * u; X[:, k+1] = x
    return X

Xz_hpic = sim_z(np.array([1., 0.]), True)
Xz_lin  = sim_z(np.array([1., 0.]), False)

print(f"  ||x(T)||  HPIC={np.linalg.norm(Xz_hpic[:,-1]):.6f}  线性PI={np.linalg.norm(Xz_lin[:,-1]):.6f}")

# ===========================================================================
# 2. Yaw 轴 HPC (偏航)
# ===========================================================================
print(f"\n>>> 2. Yaw-axis HPC")

Izz     = 0.0366
A_yaw   = np.array([[0., 1.], [0., 0.]])
B_yaw   = np.array([[0.], [1. / Izz]])
Ky_lin  = np.array([[-0.39, -0.21]])

K0_y, G0_y, P_y, mu_min_y, mu_max_y = lpc2hpc(A_yaw, B_yaw, Ky_lin)
mu_yaw = -0.5
Gd_yaw = np.eye(2) + mu_yaw * G0_y
Ky_nl  = Ky_lin - K0_y

print(f"  mu={mu_yaw}, 容许范围 [{mu_min_y:.4f}, {mu_max_y:.4f}]")
print(f"  K0_yaw={K0_y[0]}")

Ah_y, Bh_y = ZOH(DT, A_yaw, B_yaw); Bh_y = Bh_y.flatten()
Ty, Ny = 5.0, int(5.0 / DT)
hn_y = lambda x: hnorm(x, Gd_yaw, P_y)

def sim_yaw(x0, use_hpc=True):
    x = x0.copy()
    X = np.zeros((2, Ny+1)); X[:,0] = x
    for k in range(Ny):
        if use_hpc:
            u = e_hpc(x, K0_y, Ky_nl, Gd_yaw, mu_yaw, hn_y, alpha=0.1, beta=1.0)[0]
        else:
            u = (Ky_lin @ x)[0]
        x = Ah_y @ x + Bh_y * u; X[:, k+1] = x
    return X

Xy_hpc = sim_yaw(np.array([0.5, 0.]), True)
Xy_lin = sim_yaw(np.array([0.5, 0.]), False)

print(f"  ||x(T)||  HPC={np.linalg.norm(Xy_hpc[:,-1]):.6f}  线性PD={np.linalg.norm(Xy_lin[:,-1]):.6f}")

# ===========================================================================
# 3. XY 轴 HPC (水平位置+姿态, 8D)
# ===========================================================================
print(f"\n>>> 3. XY-axis HPC (8D, LQR)")

Ixx, Iyy = 0.0211, 0.0219
A_xy = np.zeros((8, 8))
A_xy[0:2, 2:4] = np.eye(2)
A_xy[2:4, 4:6] = G * np.eye(2)
A_xy[4:6, 6:8] = np.eye(2)
B_xy = np.zeros((8, 2))
B_xy[6, 0] = 1. / Iyy
B_xy[7, 1] = 1. / Ixx

Qr = np.diag([10., 10., 5., 5., 2., 2., 0.1, 0.1])
Rr = np.eye(2)
P_are = solve_continuous_are(A_xy, B_xy, Qr, Rr)
Kxy_lin = -np.linalg.inv(Rr) @ B_xy.T @ P_are

K0_xy, G0_xy, P_xy, mu_min_xy, mu_max_xy = lpc2hpc(A_xy, B_xy, Kxy_lin)
mu_xy = -1.0
Gd_xy = np.eye(8) + mu_xy * G0_xy
Kxy_nl = Kxy_lin - K0_xy

print(f"  mu={mu_xy}, 容许范围 [{mu_min_xy:.4f}, {mu_max_xy:.4f}]")
print(f"  Gd 特征值: {np.real(np.linalg.eigvals(Gd_xy))}")
print(f"  LQR K_xy =\n{Kxy_lin}")

Ah_xy, Bh_xy = ZOH(DT, A_xy, B_xy)
Txy, Nxy = 8.0, int(8.0 / DT)
hn_xy = lambda x: hnorm(x, Gd_xy, P_xy)

def sim_xy(x0, use_hpc=True):
    x = x0.copy()
    X = np.zeros((8, Nxy+1)); X[:,0] = x
    for k in range(Nxy):
        if use_hpc:
            u = e_hpc(x, K0_xy, Kxy_nl, Gd_xy, mu_xy, hn_xy, alpha=0.1, beta=1.0)
        else:
            u = Kxy_lin @ x
        x = Ah_xy @ x + Bh_xy @ u; X[:, k+1] = x
    return X

x0_xy = np.array([1., 0.5, 0., 0., 0., 0., 0., 0.])
Xxy_hpc = sim_xy(x0_xy, True)
Xxy_lin = sim_xy(x0_xy, False)

print(f"  ||x(T)||  HPC={np.linalg.norm(Xxy_hpc[:,-1]):.10f}  线性LQR={np.linalg.norm(Xxy_lin[:,-1]):.6f}")

# ===========================================================================
# 4. 绘图 — 三回路 3x3 面板, 每个面板包含线性 vs HPC 对比
# ===========================================================================
print(f"\n>>> " + L("生成对比图表...", "Generating comparison plots..."))

fig = plt.figure(figsize=(22, 15))
t_z  = np.linspace(0, Tz, Nz+1)
t_y  = np.linspace(0, Ty, Ny+1)
t_xy = np.linspace(0, Txy, Nxy+1)

# -- Row 1: Z axis --
ax = fig.add_subplot(3, 3, 1)
ax.plot(t_z, Xz_lin[0,:], 'r--', lw=1.5, label=L('线性PI ez','Linear PI ez'))
ax.plot(t_z, Xz_hpic[0,:], 'r', lw=2, label=L('HPIC ez','HPIC ez'))
ax.plot(t_z, Xz_lin[1,:], 'b--', lw=1.5, label=L('线性PI vz','Linear PI vz'))
ax.plot(t_z, Xz_hpic[1,:], 'b', lw=2, label=L('HPIC vz','HPIC vz'))
ax.axhline(0, color='k', ls='--', alpha=0.2)
ax.set_xlabel('t [s]'); ax.set_ylabel(L('状态','State'))
ax.set_title(L('Z轴 状态 (HPIC vs 线性PI)', 'Z States (HPIC vs Linear PI)'))
ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 2)
ax.plot(t_z, (Kz_lin @ Xz_lin[:2,:]).flatten(), 'g--', lw=1.5, label=L('线性','Linear'))
# Reconstruct HPIC control for plotting
uz_hpic = np.zeros(Nz+1); vi_tmp = 0.
for k in range(Nz+1):
    uh, ui = e_hpic(Xz_hpic[:,k], K0_z, Kz_nl, Ki_new, Gd_z, mu_z, hn_z, alpha=0.1)
    uz_hpic[k] = uh[0] + vi_tmp
    if k < Nz: vi_tmp += DT * ui[0]
ax.plot(t_z, uz_hpic, 'g', lw=2, label='HPIC')
ax.set_xlabel('t [s]'); ax.set_ylabel(L('推力 [m/s²]','Thrust [m/s²]'))
ax.set_title(L('Z轴 控制输入','Z Control Input'))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 3)
e_z_lin  = np.sqrt(np.sum(Xz_lin**2, axis=0))
e_z_hpic = np.sqrt(np.sum(Xz_hpic**2, axis=0))
ax.semilogy(t_z, e_z_lin, 'r--', lw=1.5, label=L('线性PI','Linear PI'))
ax.semilogy(t_z, e_z_hpic, 'r', lw=2, label='HPIC')
ax.set_xlabel('t [s]'); ax.set_ylabel('||x||')
ax.set_title(L('Z轴 误差范数', 'Z Error Norm'))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# -- Row 2: Yaw axis --
ax = fig.add_subplot(3, 3, 4)
ax.plot(t_y, Xy_lin[0,:], 'r--', lw=1.5, label=L('线性PD e_psi','Linear PD e_psi'))
ax.plot(t_y, Xy_hpc[0,:], 'r', lw=2, label=L('HPC e_psi','HPC e_psi'))
ax.plot(t_y, Xy_lin[1,:], 'b--', lw=1.5, label=L('线性PD w_z','Linear PD w_z'))
ax.plot(t_y, Xy_hpc[1,:], 'b', lw=2, label=L('HPC w_z','HPC w_z'))
ax.axhline(0, color='k', ls='--', alpha=0.2)
ax.set_xlabel('t [s]'); ax.set_ylabel(L('状态','State'))
ax.set_title(L('Yaw轴 状态 (HPC vs 线性PD)', 'Yaw States (HPC vs Linear PD)'))
ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 5)
ax.plot(t_y, (Ky_lin @ Xy_lin[:2,:]).flatten(), 'g--', lw=1.5, label=L('线性','Linear'))
ax.plot(t_y, (Ky_lin @ Xy_hpc[:2,:]).flatten(), 'g', lw=2, label='HPC')
ax.set_xlabel('t [s]'); ax.set_ylabel(L('力矩 [N.m]','Torque [N.m]'))
ax.set_title(L('Yaw轴 控制输入','Yaw Control Input'))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 6)
ax.semilogy(t_y, np.sqrt(np.sum(Xy_lin**2, axis=0)), 'r--', lw=1.5, label=L('线性PD','Linear PD'))
ax.semilogy(t_y, np.sqrt(np.sum(Xy_hpc**2, axis=0)), 'r', lw=2, label='HPC')
ax.set_xlabel('t [s]'); ax.set_ylabel('||x||')
ax.set_title(L('Yaw轴 误差范数', 'Yaw Error Norm'))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# -- Row 3: XY axis --
ax = fig.add_subplot(3, 3, 7)
ax.plot(t_xy, Xxy_lin[0,:], 'r--', lw=1.5, label=L('LQR ex','LQR ex'))
ax.plot(t_xy, Xxy_hpc[0,:], 'r', lw=2, label=L('HPC ex','HPC ex'))
ax.plot(t_xy, Xxy_lin[1,:], 'b--', lw=1.5, label=L('LQR ey','LQR ey'))
ax.plot(t_xy, Xxy_hpc[1,:], 'b', lw=2, label=L('HPC ey','HPC ey'))
ax.axhline(0, color='k', ls='--', alpha=0.2)
ax.set_xlabel('t [s]'); ax.set_ylabel(L('位置误差 [m]','Position error [m]'))
ax.set_title(L('XY轴 位置误差 (HPC vs LQR)', 'XY Position Error (HPC vs LQR)'))
ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 8)
ax.plot(t_xy, Xxy_lin[4,:], 'r--', lw=1.5, label=L('LQR theta','LQR theta'))
ax.plot(t_xy, Xxy_hpc[4,:], 'r', lw=2, label=L('HPC theta','HPC theta'))
ax.plot(t_xy, Xxy_lin[5,:], 'b--', lw=1.5, label=L('LQR phi','LQR phi'))
ax.plot(t_xy, Xxy_hpc[5,:], 'b', lw=2, label=L('HPC phi','HPC phi'))
ax.axhline(0, color='k', ls='--', alpha=0.2)
ax.set_xlabel('t [s]'); ax.set_ylabel(L('姿态角 [rad]','Attitude [rad]'))
ax.set_title(L('XY轴 姿态角 (HPC vs LQR)', 'XY Attitude (HPC vs LQR)'))
ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 9)
e_xy_lin  = np.sqrt(np.sum(Xxy_lin**2, axis=0))
e_xy_hpc  = np.sqrt(np.sum(Xxy_hpc**2, axis=0))
ax.semilogy(t_xy, e_xy_lin, 'r--', lw=1.5, label=L('线性LQR','Linear LQR'))
ax.semilogy(t_xy, e_xy_hpc, 'r', lw=2, label='HPC')
ax.set_xlabel('t [s]'); ax.set_ylabel('||x||')
ax.set_title(L('XY轴 误差范数 (对数坐标)', 'XY Error Norm (log scale)'))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

title = L('四旋翼三回路齐次控制器 — 阶跃响应对比 (虚线=线性, 实线=齐次)',
          'Quadrotor 3-Loop HPC — Step Response (dashed=linear, solid=homogeneous)')
fig.suptitle(title, fontsize=15, fontweight='bold')
plt.tight_layout()

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'uav_homogeneous_step_response.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"  " + L("图片已保存: ", "Figure saved: ") + out_png)

try:
    plt.show()
except Exception:
    pass

# ===========================================================================
# 5. 结果汇总
# ===========================================================================
print("\n" + "=" * 65)
print(L("仿真结果汇总", "Simulation Results Summary"))
print("=" * 65)
print(f"  {'Z   回路 (HPIC, mu='+str(mu_z)+')':30s} ||x(T)|| = {np.linalg.norm(Xz_hpic[:,-1]):.6f}  (线性PI: {np.linalg.norm(Xz_lin[:,-1]):.6f})")
print(f"  {'Yaw 回路 (HPC,  mu='+str(mu_yaw)+')':30s} ||x(T)|| = {np.linalg.norm(Xy_hpc[:,-1]):.6f}  (线性PD: {np.linalg.norm(Xy_lin[:,-1]):.6f})")
print(f"  {'XY  回路 (HPC,  mu='+str(mu_xy)+')':30s} ||x(T)|| = {np.linalg.norm(Xxy_hpc[:,-1]):.10f}  (LQR:   {np.linalg.norm(Xxy_lin[:,-1]):.6f})")
print()
print(L("三回路齐次控制器参数:", "Three-Loop HPC Parameters:"))
print(f"  Z:   K0={K0_z[0]}")
print(f"       Ki_new={Ki_new[0]}")
print(f"       Gd eigenvalues: {np.real(np.linalg.eigvals(Gd_z))}")
print(f"  Yaw: K0={K0_y[0]}")
print(f"       Gd eigenvalues: {np.real(np.linalg.eigvals(Gd_yaw))}")
print(f"  XY:  mu={mu_xy}, Gd eigenvalues: {np.real(np.linalg.eigvals(Gd_xy))}")
print(f"       LQR K_xy (2x8) =\n{Kxy_lin}")
