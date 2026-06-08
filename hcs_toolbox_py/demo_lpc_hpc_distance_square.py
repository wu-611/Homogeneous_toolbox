"""
demo_lpc_hpc_distance_square.py — HPC-based pursuit-evasion with formation switching

Reproduces the MATLAB lpc_hpc_distance_square.m using the Python HCS Toolbox.

Scenario:
  - Two double-integrator agents (mass m=2) move in 2D:
      Agent 1 (target):   follows a sinusoidal reference  u1 = -(x+v) + [sin(t); cos(t)]
      Agent 2 (pursuer):  tracks Agent 1 at a formation offset d using HPC

  - 4 formation points are placed on a circle of radius 1 around Agent 1.
    Agent 2 selects the closest point as its tracking offset.

  - Switching rule: when another formation point becomes closer (by tolerance
    tol=0.1) than the current one, the offset and controller are updated.

  - The linear gain for Agent 2 adapts based on the instantaneous error
    (velocity-to-position ratio), then is upgraded to HPC via lpc2hpc().

Run:
    python demo_lpc_hpc_distance_square.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hcs_toolbox_py import lpc2hpc, hnorm


# ===========================================================================
# 1. System Model — Double Integrator in 2D
#    State: [x, y, vx, vy]'
#    dx/dt = A*x + B*u,   u = [Fx/m, Fy/m]'
# ===========================================================================
m = 2.0                           # Mass [kg]
A = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [np.zeros((2, 2)), np.zeros((2, 2))]
])
B = np.vstack([
    np.zeros((2, 2)),
    np.eye(2) / m
])

# ===========================================================================
# 2. Simulation Parameters
# ===========================================================================
t = 0.0
Tmax = 30.0                        # Simulation duration [s]
h = 0.01                           # Sampling period [s]
tol_switch = 0.1                   # Switching tolerance

# Initial states
x1 = np.array([1.0, 0.0, 0.0, 0.0])   # Agent 1 (target)
x2 = np.array([5.0, 1.0, 0.0, 0.0])   # Agent 2 (pursuer)

# ===========================================================================
# 3. Formation Points — 4 points on a circle of radius 1
# ===========================================================================
m_p = 4
radius = 1.0
dl_list = []                         # List of offset vectors [dx; dy; 0; 0]
for i in range(m_p):
    angle = 2.0 * np.pi * i / m_p
    d_vec = -radius * np.array([np.cos(angle), np.sin(angle), 0.0, 0.0])
    dl_list.append(d_vec)

# ===========================================================================
# 4. Initial Formation Selection
#    Find the closest formation point to the initial error
# ===========================================================================
dist_list = []
for i in range(m_p):
    dist_list.append(np.linalg.norm(x2 - x1 - dl_list[i]))
min_idx = np.argmin(dist_list)
d = dl_list[min_idx].copy()          # Current formation offset

# ===========================================================================
# 5. Initial Linear Gain Design (adaptive, based on error)
# ===========================================================================
e = x2 - x1 - d                      # Tracking error

# Adaptive gains: a, b scale with velocity-to-position error ratio
a = max(-m * e[2] / e[0], 1.0)      # x-channel
b = max(-m * e[3] / e[1], 1.0)      # y-channel
Lambda = np.diag([a, b])

k2 = -2.0 * Lambda                   # Velocity feedback gain (2x2)
k1 = Lambda @ (k2 + Lambda) / m     # Position feedback gain (2x2)
k_lin = np.hstack([k1, k2])          # Full state feedback (2x4)

# ---- Upgrade linear gain to HPC ----
K0, G0, P, nu_min, nu_max = lpc2hpc(A, B, k_lin)
nu = nu_min                          # Homogeneity degree
Gd = np.eye(4) + nu * G0             # Generator of dilation

print("=" * 60)
print("demo_lpc_hpc_distance_square — HPC Pursuit-Evasion Tracking")
print("=" * 60)
print(f"Initial nu (homogeneity degree): {nu:.6f}")
print(f"nu_min = {nu_min:.6f}, nu_max = {nu_max:.6f}")
print(f"Initial Lambda = diag([{a:.4f}, {b:.4f}])")
print(f"Initial offset d = [{d[0]:.4f}, {d[1]:.4f}]")

# ===========================================================================
# 6. Simulation Loop
# ===========================================================================
print("Running numerical simulation...")

tl   = []      # Time log
xl1  = []      # Agent 1 state log
ul1  = []      # Agent 1 control log
xl2  = []      # Agent 2 state log
ul2  = []      # Agent 2 control log
el   = []      # Error log
lambdal = []   # Lambda gain log

while t < Tmax:
    # --- Agent 1 (target): linear controller + sinusoidal reference ---
    # u1 = -[I  I]*x1 + [sin(t); cos(t)]  (PD + sinusoidal motion)
    u1 = -np.hstack([np.eye(2), np.eye(2)]) @ x1 + np.array([np.sin(t), np.cos(t)])

    # Agent 1 dynamics (explicit Euler)
    x1 = x1 + h * (A @ x1 + B @ u1)

    # --- Agent 2 (pursuer): HPC tracking ---
    e = x2 - x1 - d                                    # Tracking error
    nx = hnorm(e, Gd, P)                              # Homogeneous norm of error

    # Clamp homogeneous norm to [0.1, 1.0]
    nx_sat = max(min(1.0, nx), 0.1)

    # HPC control law (specialized form for this demo):
    # u2 = nx_sat^(1+nu) * k_lin * expm(Gd*(1-log(nx_sat))) * e
    u2 = nx_sat ** (1.0 + nu) * k_lin @ expm(Gd * (1.0 - np.log(nx_sat))) @ e

    # Agent 2 dynamics (explicit Euler)
    x2 = x2 + h * (A @ x2 + B @ u2)

    # --- Formation switching check ---
    # If another formation point is closer (by tolerance), switch to it
    dist_list = []
    for i in range(m_p):
        dist_list.append(np.linalg.norm(x2 - x1 - dl_list[i]))
    min_dist = np.min(dist_list)
    min_idx = np.argmin(dist_list)

    if min_dist + tol_switch < np.linalg.norm(x2 - x1 - d):
        # Switch to the closer formation point
        d = dl_list[min_idx].copy()

        # Recompute error and adaptive gain
        e = x2 - x1 - d
        a = max(-m * e[2] / e[0], 4.0)    # Note: min value is 4 after switching
        b = max(-m * e[3] / e[1], 4.0)
        Lambda = np.diag([a, b])
        k2 = -2.0 * Lambda
        k1 = Lambda @ (k2 + Lambda) / m
        k_lin = np.hstack([k1, k2])

        # Re-upgrade to HPC with new gain
        K0, G0, P, nu_min, nu_max = lpc2hpc(A, B, k_lin)
        nu = nu_min
        Gd = np.eye(4) + nu * G0

    # --- Logging ---
    t += h
    tl.append(t)
    xl1.append(x1.copy())
    ul1.append(u1.copy())
    xl2.append(x2.copy())
    ul2.append(u2.copy())
    el.append(e.copy())
    lambdal.append(np.diag(Lambda).copy())

tl   = np.array(tl)
xl1  = np.array(xl1).T     # (4, N)
ul1  = np.array(ul1).T     # (2, N)
xl2  = np.array(xl2).T     # (4, N)
ul2  = np.array(ul2).T     # (2, N)
el   = np.array(el).T      # (4, N)

# Error norm ||e|| over time (first 3000 steps)
E_norm = np.sqrt(el[0, :3000]**2 + el[1, :3000]**2)

print(f"Done! Final time: {t:.2f}s")
print(f"Final ||e|| = {np.linalg.norm(x2 - x1 - d):.4f}")

# ===========================================================================
# 7. Plot Results (matching MATLAB figures 1-5, 7)
# ===========================================================================

# ---- Figure 1: X position vs time ----
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(tl, xl1[0, :], 'r', linewidth=2, label=r'$x_1$ (target)')
ax1.plot(tl, xl2[0, :], 'b', linewidth=2, label=r'$x_2$ (pursuer)')
ax1.set_xlim([0, Tmax])
ax1.set_xlabel(r'$t$ [s]', fontsize=18)
ax1.set_ylabel(r'$x$', fontsize=18)
ax1.legend(fontsize=16, frameon=False)
ax1.grid(True)
fig1.suptitle('Figure 1: X Position', fontsize=16, fontweight='bold')
fig1.tight_layout()

# ---- Figure 2: Y position vs time ----
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(tl, xl1[1, :], 'r', linewidth=2, label=r'$y_1$ (target)')
ax2.plot(tl, xl2[1, :], 'b', linewidth=2, label=r'$y_2$ (pursuer)')
ax2.set_xlim([0, Tmax])
ax2.set_ylim([-0.8, 1.2])
ax2.set_xlabel(r'$t$ [s]', fontsize=18)
ax2.set_ylabel(r'$y$', fontsize=18)
ax2.legend(fontsize=16, frameon=False)
ax2.grid(True)
fig2.suptitle('Figure 2: Y Position', fontsize=16, fontweight='bold')
fig2.tight_layout()

# ---- Figure 3: 2D Trajectory (Y vs X) ----
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.plot(xl1[0, :], xl1[1, :], 'r', linewidth=2, label=r'$r_1$ (target)')
ax3.plot(xl2[0, :], xl2[1, :], 'b', linewidth=2, label=r'$r_2$ (pursuer)')
ax3.set_ylim([-0.8, 1.2])
ax3.set_xlabel(r'$x$', fontsize=18)
ax3.set_ylabel(r'$y$', fontsize=18)
ax3.legend(fontsize=16, frameon=False)
ax3.grid(True)
ax3.set_aspect('equal')
fig3.suptitle('Figure 3: 2D Trajectory', fontsize=16, fontweight='bold')
fig3.tight_layout()

# ---- Figure 4: Control inputs of Agent 2 ----
fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(tl, ul2[0, :], 'r', linewidth=2, label=r'$u_x$')
ax4.plot(tl, ul2[1, :], 'b', linewidth=2, label=r'$u_y$')
ax4.set_ylim([-12, 6])
ax4.set_xlabel(r'$t$ [s]', fontsize=18)
ax4.set_ylabel(r'$u$', fontsize=18)
ax4.legend(fontsize=16, frameon=False)
ax4.grid(True)
fig4.suptitle('Figure 4: Pursuer Control Inputs', fontsize=16, fontweight='bold')
fig4.tight_layout()

# ---- Figure 5: Tracking Error components ----
fig5, ax5 = plt.subplots(figsize=(10, 5))
ax5.plot(tl, el[0, :], 'r', linewidth=2, label=r'$e_x$')
ax5.plot(tl, el[1, :], 'b', linewidth=2, label=r'$e_y$')
ax5.set_xlabel(r'$t$ [s]', fontsize=18)
ax5.set_ylabel(r'$e$', fontsize=18)
ax5.legend(fontsize=16, frameon=False)
ax5.grid(True)
fig5.suptitle('Figure 5: Tracking Error Components', fontsize=16, fontweight='bold')
fig5.tight_layout()

# ---- Figure 7: Error Norm ----
fig7, ax7 = plt.subplots(figsize=(10, 5))
ax7.plot(tl[:3000], E_norm, 'r', linewidth=2, label=r'$\|e\|$ (distance error)')
ax7.set_ylim([0, 3.5])
ax7.set_xlabel(r'$t$ [s]', fontsize=18)
ax7.set_ylabel(r'$\|e\|$', fontsize=18)
ax7.legend(fontsize=16, frameon=False)
ax7.grid(True)
fig7.suptitle('Figure 7: Tracking Error Norm', fontsize=16, fontweight='bold')
fig7.tight_layout()

# ---- Save figures ----
demo_dir = os.path.dirname(os.path.abspath(__file__))
for i, (fig, name) in enumerate([
    (fig1, 'lpc_hpc_dist_sq_fig1_x.png'),
    (fig2, 'lpc_hpc_dist_sq_fig2_y.png'),
    (fig3, 'lpc_hpc_dist_sq_fig3_traj.png'),
    (fig4, 'lpc_hpc_dist_sq_fig4_u.png'),
    (fig5, 'lpc_hpc_dist_sq_fig5_e.png'),
    (fig7, 'lpc_hpc_dist_sq_fig7_enorm.png'),
]):
    fpath = os.path.join(demo_dir, name)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    print(f"Figure {['1','2','3','4','5','7'][i]} saved: {fpath}")

try:
    plt.show()
except Exception:
    pass

print("\nDemo completed.")
print(f"\nKey features:")
print(f"  - 2-agent pursuit-evasion with double-integrator dynamics")
print(f"  - {m_p} formation points on a circle (radius={radius})")
print(f"  - Adaptive linear gain → HPC upgrade via lpc2hpc()")
print(f"  - Formation switching when another point is closer by tol={tol_switch}")
print(f"  - Homogeneous norm clamped to [0.1, 1.0] for bounded control")
