"""
demo_lpc2hpc.py — Example of upgrading a linear controller to HPC and FHPC

This script reproduces the MATLAB demo_lpc2hpc.m using the Python HCS Toolbox.
The system is a linearized model of a rotary inverted pendulum (QUBE-Servo 2).

System:  dx/dt = A*x + B*u,   x = [alpha, theta, alpha_dot, theta_dot]'

  where:
    alpha  - angle of the pendulum arm (state to be stabilized)
    theta  - angle of the rotary arm
    alpha_dot, theta_dot - angular velocities

Control law (HPC):
    u = K0*x + nx^(1+mu) * K * d(-ln(nx)) * x

where nx is the canonical homogeneous norm of x.

Tuning parameters:
    alpha, beta — stay-away-from-linear bounds (alpha=beta=1 → linear control)
    mu          — homogeneity degree (mu < 0 → finite-time convergence)

Run:
    python demo_lpc2hpc.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Add parent directory to path so hcs_toolbox_py can be imported
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import toolbox functions
from hcs_toolbox_py import lpc2hpc, ZOH, e_hpc, hnorm


# ===========================================================================
# 1. Rotary Inverted Pendulum Model (QUBE-Servo 2)
# ===========================================================================

# ---- Motor parameters ----
Rm = 8.4         # Resistance [Ohm]
kt = 0.042        # Current-torque constant [N·m/A]
km = 0.042        # Back-emf constant [V·s/rad]

# ---- Rotary Arm ----
mr = 0.095        # Mass [kg]
r_arm = 0.085     # Total length [m]
Jr = mr * r_arm**2 / 3.0   # Moment of inertia about pivot [kg·m^2]
br = 1e-3         # Viscous damping coefficient [N·m·s/rad]
                  # (tuned heuristically to match QUBE-Servo 2 response)

# ---- Pendulum Link ----
mp = 0.024        # Mass [kg]
Lp = 0.129        # Total length [m]
l = Lp / 2.0      # Pendulum center of mass [m]
Jp = mp * Lp**2 / 3.0      # Moment of inertia about pivot [kg·m^2]
bp = 5e-5         # Viscous damping coefficient [N·m·s/rad]
g = 9.81          # Gravity constant [m/s^2]

# Total Inertia (coupled term)
Jt = Jr * Jp - mp**2 * r_arm**2 * l**2


# ===========================================================================
# 2. Linearized State-Space Model (upper equilibrium)
#    State: x = [x1, x2, x3, x4]'
#      x1 - pendulum arm angle
#      x2 - rotary arm angle
#      x3 - pendulum arm angular velocity
#      x4 - rotary arm angular velocity
# ===========================================================================

A = np.array([
    [0.0, 0.0,                        1.0,                   0.0],
    [0.0, 0.0,                        0.0,                   1.0],
    [0.0, mp**2 * l**2 * r_arm * g / Jt, -br * Jp / Jt,     -mp * l * r_arm * bp / Jt],
    [0.0, mp * g * l * Jr / Jt,        -mp * l * r_arm * br / Jt, -Jr * bp / Jt]
])

B = np.array([
    [0.0],
    [0.0],
    [Jp / Jt],
    [mp * l * r_arm / Jt]
])

# ---- Add actuator dynamics ----
A[2, 2] = A[2, 2] - km * km / Rm * B[2, 0]
A[3, 2] = A[3, 2] - km * km / Rm * B[3, 0]
B = km * B / Rm

# ---- Linear feedback gain (provided by manufacturer) ----
Klin = np.array([[2.0, -35.0, 1.5, -3.0]])


# ===========================================================================
# 3. HPC/FHPC Design: upgrade linear controller to homogeneous
# ===========================================================================
print("=" * 60)
print("demo_lpc2hpc — Upgrading Linear Control to HPC/FHPC")
print("=" * 60)

K0, G0, P, mu_min, mu_max = lpc2hpc(A, B, Klin)

print(f"mu_min = {mu_min:.6f},  mu_max = {mu_max:.6f}")
print(f"K0 = {K0}")
print(f"G0 eigenvalues: {np.real(np.linalg.eigvals(G0))}")

# ---- Select homogeneity degree mu ----
# mu must satisfy: mu_min <= mu <= mu_max
mu = -1.0     # Negative homogeneity degree → finite-time convergence

# Generator of dilation: Gd = I + mu * G0
Gd = np.eye(4) + mu * G0

# Nonlinear gain: K = K_lin - K0
K = Klin - K0

print(f"mu = {mu}")
print(f"Gd eigenvalues: {np.real(np.linalg.eigvals(Gd))}")

# ---- Alternative selections (commented, as in MATLAB) ----
# mu = mu_min  # HPC with negative homogeneity degree
# mu = mu_max  # HPC with positive homogeneity degree
# (For FHPC, use G0, mu_min, mu_max directly)


# ===========================================================================
# 4. Numerical Simulation
# ===========================================================================
print("Running numerical simulation...")

t = 0.0
Tmax = 3.0           # Simulation duration [s]
h = 0.001             # Sampling period [s]

# Initial state: small perturbation from equilibrium
x = np.array([1.0, 1.0, 0.0, 0.0])

# Logs
tl = [t]
xl = [x.copy()]
ul = []

# ---- Tuning parameters alpha, beta ----
# alpha = beta = 1   → linear control (no homogeneous upgrade)
# alpha → 0, beta → ∞ → full homogeneous control
alpha = 0.1    # Lower bound for homogeneous norm (close to origin)
beta  = 1.0    # Upper bound for homogeneous norm (far from origin)
# For HPC with positive degree: alpha=1, beta=100
# For FHPC: alpha=0.1, beta=100

noise = 0.0    # Measurement noise magnitude (set >0 for robustness test)

# ---- Discretize plant with Zero-Order Hold ----
Ah, Bh = ZOH(h, A, B)

# ---- Homogeneous norm function ----
hn_fun = lambda x_vec: hnorm(x_vec, Gd, P)

import warnings
warnings.filterwarnings('ignore')

while t < Tmax:
    # Simulate noisy measurements
    xm = x + 2.0 * noise * (np.random.rand(4) - 0.5)

    # --- Linear control (uncomment for comparison) ---
    # u = Klin @ xm

    # --- HPC: explicit discretization ---
    u = e_hpc(xm, K0, K, Gd, mu, hn_fun, alpha, beta)

    # --- Plant simulation with control saturation (as in QUBE-Servo 2) ---
    u_sat = np.clip(u, -10.0, 10.0)
    x = Ah @ x + Bh.flatten() * u_sat

    t += h
    tl.append(t)
    xl.append(x.copy())
    ul.append(u_sat)

ul.append(ul[-1] if ul else 0.0)  # Match MATLAB: ul=[ul u]

print("Done!")
print(f"||x(Tmax)|| = {np.linalg.norm(x):.6f}")


# ===========================================================================
# 5. Plot Results
# ===========================================================================
tl_arr = np.array(tl)
xl_arr = np.array(xl).T    # Shape (4, N)
ul_arr = np.array(ul)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left plot: States ---
ax1.plot(tl_arr, xl_arr[0, :], linewidth=2, label=r'$x_1$ (pendulum angle)')
ax1.plot(tl_arr, xl_arr[1, :], linewidth=2, label=r'$x_2$ (rotary arm angle)')
ax1.plot(tl_arr, xl_arr[2, :], linewidth=2, label=r'$x_3$ (pendulum velocity)')
ax1.plot(tl_arr, xl_arr[3, :], linewidth=2, label=r'$x_4$ (rotary velocity)')
ax1.set_xlabel(r'$t$ [s]', fontsize=14)
ax1.set_ylabel(r'$x$', fontsize=14)
ax1.set_title('States (n=4)', fontsize=14)
ax1.set_xlim([0, Tmax])
ax1.set_ylim([-5, 5])
ax1.grid(True)
ax1.legend(fontsize=11, loc='best')

# --- Right plot: Control input ---
ax2.plot(tl_arr[:len(ul_arr)], ul_arr, linewidth=2, color='red')
ax2.set_xlabel(r'$t$ [s]', fontsize=14)
ax2.set_ylabel(r'$u$', fontsize=14)
ax2.set_title('HPC, m=1', fontsize=14)
ax2.set_xlim([0, Tmax])
ax2.set_ylim([-5, 5])
ax2.grid(True)

fig.suptitle('demo\_lpc2hpc: Homogeneous Proportional Control (HPC)', fontsize=15, fontweight='bold')
plt.tight_layout()
# Save figure (comment out plt.show() if running headless)
fig.savefig(os.path.join(os.path.dirname(__file__), 'demo_lpc2hpc.png'), dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {os.path.join(os.path.dirname(__file__), 'demo_lpc2hpc.png')}")
try:
    plt.show()
except Exception:
    pass

print("\nDemo completed.")
