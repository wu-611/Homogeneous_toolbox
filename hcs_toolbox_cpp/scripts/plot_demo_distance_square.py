#!/usr/bin/env python3
"""Plot results from C++ demo_lpc_hpc_distance_square."""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "demo_lpc_hpc_distance_square_cpp.csv"
if not os.path.exists(csv_path):
    alt = os.path.join(os.path.dirname(__file__), "..", "build", "demo_lpc_hpc_distance_square_cpp.csv")
    if os.path.exists(alt):
        csv_path = alt
    else:
        print(f"Error: {csv_path} not found. Run demo_lpc_hpc_distance_square first.")
        sys.exit(1)

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
t    = data[:, 0]
x1   = data[:, 1];  y1   = data[:, 2]
vx1  = data[:, 3];  vy1  = data[:, 4]
x2   = data[:, 5];  y2   = data[:, 6]
vx2  = data[:, 7];  vy2  = data[:, 8]
ex   = data[:, 9];  ey   = data[:, 10]
u2x  = data[:, 11]; u2y  = data[:, 12]

e_norm = np.sqrt(ex**2 + ey**2)

# Figure 1: X position vs time
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(t, x1, 'r', linewidth=2, label=r'$x_1$ (target)')
ax1.plot(t, x2, 'b', linewidth=2, label=r'$x_2$ (pursuer)')
ax1.set_xlim([0, t[-1]])
ax1.set_xlabel(r'$t$ [s]', fontsize=18)
ax1.set_ylabel(r'$x$', fontsize=18)
ax1.legend(fontsize=16, frameon=False)
ax1.grid(True)
fig1.suptitle('Figure 1: X Position', fontsize=16, fontweight='bold')
fig1.tight_layout()

# Figure 2: Y position vs time
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(t, y1, 'r', linewidth=2, label=r'$y_1$ (target)')
ax2.plot(t, y2, 'b', linewidth=2, label=r'$y_2$ (pursuer)')
ax2.set_xlim([0, t[-1]])
ax2.set_ylim([-0.8, 1.2])
ax2.set_xlabel(r'$t$ [s]', fontsize=18)
ax2.set_ylabel(r'$y$', fontsize=18)
ax2.legend(fontsize=16, frameon=False)
ax2.grid(True)
fig2.suptitle('Figure 2: Y Position', fontsize=16, fontweight='bold')
fig2.tight_layout()

# Figure 3: 2D Trajectory
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.plot(x1, y1, 'r', linewidth=2, label=r'$r_1$ (target)')
ax3.plot(x2, y2, 'b', linewidth=2, label=r'$r_2$ (pursuer)')
ax3.set_ylim([-0.8, 1.2])
ax3.set_xlabel(r'$x$', fontsize=18)
ax3.set_ylabel(r'$y$', fontsize=18)
ax3.legend(fontsize=16, frameon=False)
ax3.grid(True)
ax3.set_aspect('equal')
fig3.suptitle('Figure 3: 2D Trajectory', fontsize=16, fontweight='bold')
fig3.tight_layout()

# Figure 4: Control inputs
fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(t, u2x, 'r', linewidth=2, label=r'$u_x$')
ax4.plot(t, u2y, 'b', linewidth=2, label=r'$u_y$')
ax4.set_ylim([-12, 6])
ax4.set_xlabel(r'$t$ [s]', fontsize=18)
ax4.set_ylabel(r'$u$', fontsize=18)
ax4.legend(fontsize=16, frameon=False)
ax4.grid(True)
fig4.suptitle('Figure 4: Pursuer Control Inputs', fontsize=16, fontweight='bold')
fig4.tight_layout()

# Figure 5: Tracking error components
fig5, ax5 = plt.subplots(figsize=(10, 5))
ax5.plot(t, ex, 'r', linewidth=2, label=r'$e_x$')
ax5.plot(t, ey, 'b', linewidth=2, label=r'$e_y$')
ax5.set_xlabel(r'$t$ [s]', fontsize=18)
ax5.set_ylabel(r'$e$', fontsize=18)
ax5.legend(fontsize=16, frameon=False)
ax5.grid(True)
fig5.suptitle('Figure 5: Tracking Error Components', fontsize=16, fontweight='bold')
fig5.tight_layout()

# Figure 7: Error norm
fig7, ax7 = plt.subplots(figsize=(10, 5))
ax7.plot(t, e_norm, 'r', linewidth=2, label=r'$\|e\|$')
ax7.set_ylim([0, 3.5])
ax7.set_xlabel(r'$t$ [s]', fontsize=18)
ax7.set_ylabel(r'$\|e\|$', fontsize=18)
ax7.legend(fontsize=16, frameon=False)
ax7.grid(True)
fig7.suptitle('Figure 7: Tracking Error Norm', fontsize=16, fontweight='bold')
fig7.tight_layout()

# Save all figures
out_dir = os.path.dirname(os.path.abspath(__file__))
for fig, name in [(fig1, "cpp_fig1_x"), (fig2, "cpp_fig2_y"),
                   (fig3, "cpp_fig3_traj"), (fig4, "cpp_fig4_u"),
                   (fig5, "cpp_fig5_e"), (fig7, "cpp_fig7_enorm")]:
    fpath = os.path.join(out_dir, name + ".png")
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {fpath}")

plt.show()
