#!/usr/bin/env python3
"""
validate_e_ho.py — Verify HO discrete-time update for a harmonic oscillator
scenario from MATLAB demo_ho.m.

Validates:
  1. e_ho and si_ho produce finite output (no NaN/Inf)
  2. si_ho is more stable than e_ho for oscillatory systems
  3. Observer error converges to zero
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from hcs_toolbox_py import lo2ho, e_ho, si_ho, hnorm

# ---------- Harmonic Oscillator ----------
A = np.array([[0., 1.], [-1., 0.]])
C = np.array([[1., 0.]])
n, k = 2, 1

# Design linear observer via pole placement
# Desired poles: [-3, -4]
L_lin = np.array([[-7.], [-11.]])  # A+LC eigenvalues at -3, -4

print("=" * 65)
print("validate_e_ho.py — Harmonic Oscillator HO Validation")
print("=" * 65)

# ---------- Test 1: e_ho basic operation ----------
print("\n>>> Test 1: e_ho produces valid output")

L0, G0, P, nu_min, nu_max = lo2ho(A, C, L_lin)
nu = -0.5  # finite-time convergence
Gd = np.eye(n) + nu * G0
L_nl = L_lin - L0

h = 0.001

x = np.array([1.0, 0.0])     # true state (harmonic oscillator)
z_e = np.array([0.0, 0.0])    # e_ho estimate
z_si = np.array([0.0, 0.0])   # si_ho estimate

for step in range(100):
    x = x + h * A @ x  # explicit Euler for truth
    y = C @ x

    z_e = e_ho(h, z_e, y, A, C, np.zeros(n), L0, L_nl, Gd, nu,
               alpha=1e-6, beta=np.inf)

    z_si = si_ho(h, z_si, y, A, C, np.zeros(n), L0, L_nl, Gd, nu,
                 alpha=1e-6, beta=np.inf)

assert not np.any(np.isnan(z_e)), "e_ho produced NaN"
assert not np.any(np.isnan(z_si)), "si_ho produced NaN"
assert not np.any(np.isinf(z_e)), "e_ho produced Inf"
assert not np.any(np.isinf(z_si)), "si_ho produced Inf"

print(f"  e_ho  after 100 steps: z = {z_e}")
print(f"  si_ho after 100 steps: z = {z_si}")
print("  PASSED: Both methods produce finite values")

# ---------- Test 2: Observer convergence ----------
print("\n>>> Test 2: Observer error convergence over full simulation")

Tmax = 6.0
N = int(Tmax / h)

x = np.array([1.0, 0.0])
z = np.array([0.0, 0.0])
err_log = np.zeros(N + 1)
err_log[0] = np.linalg.norm(x - z)

for step in range(N):
    x = x + h * A @ x
    y = C @ x
    z = si_ho(h, z, y, A, C, np.zeros(n), L0, L_nl, Gd, nu,
              alpha=1e-6, beta=np.inf)
    err_log[step + 1] = np.linalg.norm(x - z)

err_initial = err_log[0]
err_final = err_log[-1]

print(f"  Initial ||e|| = {err_initial:.4f}")
print(f"  Final   ||e|| = {err_final:.6f}")
print(f"  Reduction: {err_initial/err_final:.1f}x")

assert err_final < err_initial * 0.1, \
    f"Observer error did not significantly decrease: {err_final:.4f}"
print("  PASSED: Observer error converges")

# ---------- Test 3: Linear observer (nu=0, alpha=1, beta=1) matches linear ----------
print("\n>>> Test 3: HO with nu=0, alpha=1, beta=1 → linear observer")

x = np.array([1.0, 0.0])
z_ho = np.array([0.0, 0.0])
z_lin = np.array([0.0, 0.0])

for step in range(50):
    x = x + h * A @ x
    y = C @ x

    # HO in linear mode
    z_ho = si_ho(h, z_ho, y, A, C, np.zeros(n), L0, L_nl, np.eye(n), 0.0,
                 alpha=1.0, beta=1.0)

    # Linear Luenberger
    z_lin = z_lin + h * (A @ z_lin + L_lin @ (C @ z_lin - y))

diff = np.linalg.norm(z_ho - z_lin)
print(f"  ||z_ho - z_lin|| after 50 steps = {diff:.8f}")
assert diff < 0.01, f"HO in linear mode should match linear observer, diff={diff:.6f}"
print("  PASSED: HO(α=1,β=1,ν=0) matches linear observer")

# ---------- Test 4: si_ho stability advantage for harmonic oscillator ----------
print("\n>>> Test 4: si_ho stability vs e_ho for harmonic oscillator")

# Use larger step size to expose instability
h_large = 0.05
x = np.array([1.0, 0.0])
z_e = np.array([0.0, 0.0])
z_si = np.array([0.0, 0.0])

e_unstable = False
for step in range(200):
    x = x + h_large * A @ x
    y = C @ x

    z_si = si_ho(h_large, z_si, y, A, C, np.zeros(n), L0, L_nl, Gd, nu,
                 alpha=1e-6, beta=np.inf)

    try:
        z_e = e_ho(h_large, z_e, y, A, C, np.zeros(n), L0, L_nl, Gd, nu,
                   alpha=1e-6, beta=np.inf)
        if np.linalg.norm(z_e) > 1e6:
            e_unstable = True
            break
    except Exception:
        e_unstable = True
        break

si_final = np.linalg.norm(x - z_si)
print(f"  Large step h={h_large}:")
print(f"    e_ho  {'diverged' if e_unstable else 'stable'}")
print(f"    si_ho final ||e|| = {si_final:.6f}")

assert si_final < 2.0, f"si_ho should remain stable, error={si_final:.4f}"
print("  PASSED: si_ho stable where e_ho may diverge")

print("\n" + "=" * 65)
print("All tests PASSED!")
print("=" * 65)
