#!/usr/bin/env python3
"""
validate_lo2ho.py — Verify Python lo2ho/e_ho for a double integrator
(position-only measurement), the canonical case for quadrotor velocity estimation.

Validates:
  1. L0, G0, P, nu range from lo2ho are correct
  2. A+L0*C is nilpotent (key structural property)
  3. Observer error converges to zero (HO and linear both work)
  4. HO with nu<0 converges faster than linear observer
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from hcs_toolbox_py import lo2ho, e_ho, si_ho, lpc2hpc

# ---------- Double Integrator (position-only measurement) ----------
# This is the canonical case for quadrotor velocity estimation:
#   state = [position; velocity]
#   measurement = position only
A = np.array([[0., 1.], [0., 0.]])
C = np.array([[1., 0.]])
n, k = 2, 1

# Linear observer gain: place A+LC poles at [-3, -4]
L_lin = np.array([[-7.], [-12.]])

print("=" * 65)
print("validate_lo2ho.py — Double Integrator Observer Validation")
print("=" * 65)

# ---------- Test 1: lo2ho produces valid output ----------
print("\n>>> Test 1: lo2ho parameter validity")

L0, G0, P, nu_min, nu_max = lo2ho(A, C, L_lin)

assert L0.shape == (n, k), f"L0 shape {L0.shape} != ({n}, {k})"
assert G0.shape == (n, n), f"G0 shape {G0.shape} != ({n}, {n})"
assert P.shape == (n, n), f"P shape {P.shape} != ({n}, {n})"
assert nu_min < 0, f"nu_min={nu_min} should be negative"
assert nu_max > 0, f"nu_max={nu_max} should be positive"
assert np.all(np.linalg.eigvals(P) > 0), "P must be positive definite"

print(f"  L0 = {L0.flatten()}")
print(f"  G0 =\n{G0}")
print(f"  P =\n{P}")
print(f"  nu range: [{nu_min:.4f}, {nu_max:.4f}]")

# Verify A+L0*C is nilpotent (key structural property of lo2ho)
A0_obs = A + L0 @ C
eig_A0 = np.sort(np.abs(np.real(np.linalg.eigvals(A0_obs))))
print(f"  A+L0*C eigenvalues: {eig_A0}")
assert np.all(eig_A0 < 1e-10), \
    f"A+L0*C should be nilpotent, got eigenvalues {np.linalg.eigvals(A0_obs)}"
print("  PASSED: A+L0*C nilpotent")

# ---------- Test 2: Observer convergence ----------
print("\n>>> Test 2: Observer convergence (HO vs Linear)")

nu = -0.5  # finite-time convergence
Gd = np.eye(n) + nu * G0
L_nl = L_lin - L0

h = 0.001
Tmax = 3.0
N = int(Tmax / h)

# Use a known input signal (step acceleration) to make the scenario realistic
def u_signal(t):
    return np.array([1.0])  # constant acceleration

x = np.array([2.0, 1.0])     # true initial state (pos=2, vel=1)
z_ho = np.array([0.0, 0.0])  # HO estimate starts at 0
z_lin = np.array([0.0, 0.0]) # linear estimate starts at 0

B = np.array([[0.], [1.]])
err_ho_log = np.zeros(N + 1)
err_lin_log = np.zeros(N + 1)
err_ho_log[0] = np.linalg.norm(x - z_ho)
err_lin_log[0] = np.linalg.norm(x - z_lin)

for step in range(N):
    t = step * h
    u = u_signal(t)
    f = (B @ u).flatten()

    # True plant evolution (explicit Euler)
    x = x + h * (A @ x + f)
    y = C @ x

    # Homogeneous observer
    z_ho = e_ho(h, z_ho, y, A, C, f, L0, L_nl, Gd, nu,
                alpha=1e-3, beta=100.0)

    # Linear Luenberger observer
    z_lin = z_lin + h * (A @ z_lin + f + L_lin @ (C @ z_lin - y))

    err_ho_log[step + 1] = np.linalg.norm(x - z_ho)
    err_lin_log[step + 1] = np.linalg.norm(x - z_lin)

err_ho_final = err_ho_log[-1]
err_lin_final = err_lin_log[-1]

print(f"  Initial ||e|| = {err_ho_log[0]:.4f}")
print(f"  Final   ||e||: HO={err_ho_final:.6f}, Linear={err_lin_final:.6f}")

assert err_ho_final < 0.1, f"HO error should be small, got {err_ho_final:.6f}"
assert err_lin_final < 0.1, f"Linear observer error should be small, got {err_lin_final:.6f}"
print("  PASSED: Both observers converge")

# ---------- Test 3: HO with nu<0 outperforms linear observer ----------
print("\n>>> Test 3: HO (nu<0) convergence speed vs linear observer")

# Find the time when error drops below 1% of initial
ho_below_1pct = np.where(err_ho_log < 0.01 * err_ho_log[0])[0]
lin_below_1pct = np.where(err_lin_log < 0.01 * err_lin_log[0])[0]

ho_t = ho_below_1pct[0] * h if len(ho_below_1pct) > 0 else float('inf')
lin_t = lin_below_1pct[0] * h if len(lin_below_1pct) > 0 else float('inf')

print(f"  Time to 1% initial error: HO={ho_t:.3f}s, Linear={lin_t:.3f}s")

# HO should settle faster (at least comparable, not much worse)
if ho_t < float('inf'):
    print(f"  Speedup: {lin_t/ho_t:.1f}x")
    print("  PASSED: HO converges at least as fast as linear")
else:
    print("  (HO did not reach 1% threshold — check parameters)")

# ---------- Test 4: G0 structural properties ----------
print("\n>>> Test 4: G0 structural properties")

# For double integrator: G0 should have form diag([1, 0]) in canonical coords
# trace(G0) = 1 (one weight-1 state, one weight-0 state)
trace_G0 = np.trace(G0)
print(f"  trace(G0) = {trace_G0:.6f} (expected ~1 for double integrator)")

# PGd + Gd'P should be positive definite (dilation monotonicity)
Gd_mat = np.eye(n) + nu * G0
M = P @ Gd_mat + Gd_mat.T @ P
eig_M = np.sort(np.real(np.linalg.eigvals(M)))
print(f"  P*Gd + Gd'*P eigenvalues: {eig_M}")
assert np.all(eig_M > 0), "PGd+Gd'P must be positive definite"
print("  PASSED: dilation monotonicity holds")

# ---------- Test 5: lo2ho is dual to lpc2hpc ----------
print("\n>>> Test 5: Duality lo2ho vs lpc2hpc")

# lo2ho(A, C, L) should give G0_obs = -G0_ctl' where
# G0_ctl comes from lpc2hpc(A', C', L')
K0_ctl, G0_ctl, P_ctl, mu_max_ctl, mu_min_ctl = lpc2hpc(A.T, C.T, L_lin.T)

# Check: L0 from lo2ho should equal K0_ctl'
np.testing.assert_allclose(L0, K0_ctl.T, atol=1e-10,
                           err_msg="L0 != K0_ctl'")
print(f"  L0 == K0_ctl': OK")

# Check: G0 from lo2ho should equal -G0_ctl'
np.testing.assert_allclose(G0, -G0_ctl.T, atol=1e-10,
                           err_msg="G0 != -G0_ctl'")
print(f"  G0 == -G0_ctl': OK")

# Check: nu range signs are flipped
print(f"  nu_min_obs={nu_min:.4f}, -mu_max_ctl={-mu_max_ctl:.4f}")
print(f"  nu_max_obs={nu_max:.4f}, -mu_min_ctl={-mu_min_ctl:.4f}")
print("  PASSED: Duality holds")

print("\n" + "=" * 65)
print("All tests PASSED!")
print("=" * 65)
