#!/usr/bin/env python3
"""Plot results from C++ demo_lpic2hpic."""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "demo_lpic2hpic_cpp.csv"
if not os.path.exists(csv_path):
    alt = os.path.join(os.path.dirname(__file__), "..", "build", "demo_lpic2hpic_cpp.csv")
    if os.path.exists(alt):
        csv_path = alt
    else:
        print(f"Error: {csv_path} not found. Run demo_lpic2hpic first.")
        sys.exit(1)

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
t   = data[:, 0]
x1  = data[:, 1]
x2  = data[:, 2]
x3  = data[:, 3]
x4  = data[:, 4]
u   = data[:, 5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(t, x1, linewidth=2, label=r'$x_1$ (pendulum angle)')
ax1.plot(t, x2, linewidth=2, label=r'$x_2$ (rotary arm angle)')
ax1.plot(t, x3, linewidth=2, label=r'$x_3$ (pendulum velocity)')
ax1.plot(t, x4, linewidth=2, label=r'$x_4$ (rotary velocity)')
ax1.set_xlabel(r'$t$ [s]', fontsize=14)
ax1.set_ylabel(r'$x$', fontsize=14)
ax1.set_title('States (n=4)', fontsize=14)
ax1.set_xlim([0, t[-1]])
ax1.grid(True)
ax1.legend(fontsize=11, loc='best')

ax2.plot(t, u, linewidth=2, color='red')
ax2.set_xlabel(r'$t$ [s]', fontsize=14)
ax2.set_ylabel(r'$u$', fontsize=14)
ax2.set_title('HPIC Control Input (m=1)', fontsize=14)
ax2.set_xlim([0, t[-1]])
ax2.grid(True)

fig.suptitle('demo\_lpic2hpic (C++): Homogeneous PI Control', fontsize=15, fontweight='bold')
plt.tight_layout()

out = os.path.splitext(csv_path)[0] + ".png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()
