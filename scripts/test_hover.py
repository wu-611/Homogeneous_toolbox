#!/usr/bin/env python3
"""Quick hover test to debug HPC controller stability."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.quadrotor_se3 import QuadrotorSE3
from controllers.se3_homogeneous_full import SE3HomogeneousController, LeeGeometricPD

m = 1.4
J = np.diag([0.0211, 0.0219, 0.0366])
g = 9.81
model = QuadrotorSE3(m, J, g)
dt = 0.001

# Test: hovering at origin (pos_d = [0,0,0])
pos_d = np.array([0.0, 0.0, 0.0])
vel_d = np.array([0.0, 0.0, 0.0])
yaw_d = 0.0

# Start slightly off hover
pos0 = np.array([0.1, 0.0, -0.1])
vel0 = np.zeros(3)
R0 = np.eye(3)
omega0 = np.zeros(3)
state = model.make_state(pos0, vel0, R0, omega0)

# Test with Lee PD first (known to work)
print("Testing Lee PD...")
lee = LeeGeometricPD(m, J, g, kx=16.0, kv=8.0, kR=12.0, komega=6.0)
for k in range(200):
    u = lee.compute_control(state, pos_d, vel_d, yaw_d)
    state = model.step_rk4(state, u, dt)
pos, vel, R, omega = model.unpack_state(state)
print(f"  After 0.2s: pos={pos}, thrust={u[0]:.2f}, M={u[1:4]}")

# Test HPC mu=0
print("\nTesting HPC mu=0...")
state = model.make_state(pos0, vel0, R0, omega0)
hpc = SE3HomogeneousController(m, J, g, mu_p=0.0, mu_a=0.0)
for k in range(200):
    u = hpc.compute_control(state, pos_d, vel_d, yaw_d)
    state = model.step_rk4(state, u, dt)
pos, vel, R, omega = model.unpack_state(state)
print(f"  After 0.2s: pos={pos}, thrust={u[0]:.2f}, M={u[1:4]}")

# Debug the HPC intermediate values
state = model.make_state(pos0, vel0, R0, omega0)
debug = hpc.get_debug_info(state, pos_d, vel_d, yaw_d)
print(f"  Debug: e_pos={debug['e_pos']}")
print(f"  Debug: e_vel={debug['e_vel']}")
print(f"  Debug: u_pos={debug['u_pos']}")
print(f"  Debug: F_des={debug['F_des']}")
print(f"  Debug: b3_des={debug['b3_des']}")
print(f"  Debug: theta_e={debug['theta_e']}")

# Single step
u = hpc.compute_control(state, pos_d, vel_d, yaw_d)
print(f"  Single step: thrust={u[0]:.4f}, tau={u[1:4]}")
state2 = model.step_rk4(state, u, dt)
pos2, vel2, R2, omega2 = model.unpack_state(state2)
print(f"  Next state: pos={pos2}, vel={vel2}")

# Check what Lee PD would output for same state
u_lee = lee.compute_control(state, pos_d, vel_d, yaw_d)
print(f"\n  Lee PD for same state: thrust={u_lee[0]:.4f}, tau={u_lee[1:4]}")
print(f"  Ratio HPC/Lee: thrust={u[0]/u_lee[0]:.4f}, tau_x={u[1]/u_lee[1]:.4f}")
