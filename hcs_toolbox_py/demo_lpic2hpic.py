"""
demo_lpic2hpic.py — Example of upgrading a linear PI controller to HPIC/FHPIC

This script reproduces the MATLAB demo_lpic2hpic.m using the Python HCS Toolbox.
The system is a linearized model of a rotary inverted pendulum (QUBE-Servo 2).

System:  dx/dt = A*x + B*u,   x = [alpha, theta, alpha_dot, theta_dot]'

Control law (HPIC — Homogeneous Proportional-Integral Controller):
    u = uh + v,    dv/dt = ui

    uh = K0*x + nx^(1+mu) * K * d(-ln(nx)) * x     (proportional component)
    ui = nx^(1+2*mu) * Ki * d(-ln(nx)) * x          (integral component)

The LPIC → HPIC upgrade uses lpic2hpic(), which redesigns the integral gain
for the homogeneous setting.

Tuning parameters:
    alpha, beta — stay-away-from-linear bounds
    mu          — homogeneity degree (mu < 0 → finite-time convergence)

A constant perturbation p is added to test integral action (disturbance rejection).

Run:
    python demo_lpic2hpic.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so hcs_toolbox_py can be imported
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import toolbox functions
from hcs_toolbox_py import lpic2hpic, e_hpic, hnorm
# Note: e_fhpic is available for Fixed-time HPIC comparison


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

# ---- Pendulum Link ----
mp = 0.024        # Mass [kg]
Lp = 0.129        # Total length [m]
l = Lp / 2.0      # Pendulum center of mass [m]
Jp = mp * Lp**2 / 3.0      # Moment of inertia about pivot [kg·m^2]
bp = 5e-5         # Viscous damping coefficient [N·m·s/rad]
g = 9.81          # Gravity constant [m/s^2]

# Total Inertia
Jt = Jr * Jp - mp**2 * r_arm**2 * l**2


# ===========================================================================
# 2. Linearized State-Space Model (upper equilibrium)
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


# ===========================================================================
# 3. HPIC/FHPIC Design: upgrade linear PI controller to HPIC
# ===========================================================================
print("=" * 60)
print("demo_lpic2hpic — Upgrading Linear PI Control to HPIC/FHPIC")
print("=" * 60)

# ---- Linear feedback gains (provided by manufacturer) ----
Klin = np.array([[2.0, -35.0, 1.5, -3.0]])       # Proportional gain
Klin_int = np.array([[0.5, -26.66, 1.26, -2.73]])  # Integral gain

# ---- Upgrade LPIC to HPIC ----
K0, G0, P, Ki_new, mu1, mu2 = lpic2hpic(A, B, Klin, Klin_int)

print(f"mu_min = {mu1:.6f},  mu_max = {mu2:.6f}")
print(f"K0     = {K0}")
print(f"Ki_new = {Ki_new}")

# ---- Select homogeneity degree mu ----
mu = 0.16    # Negative homogeneity degree → finite-time convergence
# Alternative: mu = mu2  (positive degree for nearly fixed-time)

# Generator of dilation: Gd = I + mu * G0
Gd = np.eye(4) + mu * G0

# Nonlinear proportional gain: K = K_lin - K0
K = Klin - K0

print(f"mu = {mu}")
print(f"Gd eigenvalues: {np.real(np.linalg.eigvals(Gd))}")


# ===========================================================================
# 4. Numerical Simulation
# ===========================================================================
print("Running numerical simulation...")

t = 0.0
Tmax = 4.0           # Simulation duration [s]
h = 0.001             # Sampling period [s]

# Initial state
x = np.array([1.0, 1.0, 0.0, 0.0])

# Logs
tl = [t]
xl = [x.copy()]
ul = []

# ---- Tuning parameters alpha, beta ----
# alpha = beta = 1 → linear control (no homogeneous upgrade)
alpha = 0.001   # Lower bound (close to origin)
beta  = 1000.0   # Upper bound (far from infinity)
# For FHPC/FHPIC with global upgrade: alpha=0.01, beta=2

noise = 0.0     # Measurement noise magnitude

# ---- Integral accumulator ----
v = 0.0

# ---- Constant perturbation (matched, with unknown bound) ----
p = 1.0         # Constant disturbance — tests integral action

# ---- Homogeneous norm function ----
hn_fun = lambda x_vec: hnorm(x_vec, Gd, P)

import warnings
warnings.filterwarnings('ignore')

while t < Tmax:
    # Simulate noisy measurements
    xm = x + 2.0 * noise * (np.random.rand(4) - 0.5)

    # --- Linear PI control (uncomment for comparison) ---
    # u_lin = Klin @ xm
    # ui_lin = Ki_new @ xm

    # --- HPIC: homogeneous PI control ---
    uh, ui = e_hpic(xm, K0, K, Ki_new, Gd, mu, hn_fun, alpha, beta)

    # --- FHPIC (uncomment for comparison) ---
    # from hcs_toolbox_py import e_fhpic
    # uh, ui = e_fhpic(xm, K0, K, Ki_new, G0, mu1, mu2, P, alpha, beta)

    # Total control = proportional + integral
    u = uh + v
    v = v + h * ui    # Integrate: dv/dt = ui

    # --- Linear control (uncomment for comparison) ---
    # u = Klin @ xm

    # --- Plant: explicit Euler method with perturbation ---
    x = x + h * A @ x + h * B.flatten() * (u + p)

    t += h
    tl.append(t)
    xl.append(x.copy())
    ul.append(u)

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
ax2.set_title('HPIC (Proportional + Integral), m=1', fontsize=14)
ax2.set_xlim([0, Tmax])
ax2.set_ylim([-5, 5])
ax2.grid(True)

fig.suptitle('demo\_lpic2hpic: Homogeneous PI Control (HPIC) with Constant Disturbance', fontsize=15, fontweight='bold')
plt.tight_layout()
# Save figure (comment out plt.show() if running headless)
fig.savefig(os.path.join(os.path.dirname(__file__), 'demo_lpic2hpic.png'), dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {os.path.join(os.path.dirname(__file__), 'demo_lpic2hpic.png')}")
try:
    plt.show()
except Exception:
    pass

print("\nDemo completed.")
print(f"\nKey observation:")
print(f"  - With perturbation p={p}, linear PI would have steady-state error.")
print(f"  - HPIC integral action nx^(1+2*mu)*Ki*d(-ln(nx))*x compensates the disturbance.")
print(f"  - Homogeneity degree mu={mu} enables finite-time/nearly fixed-time convergence.")
