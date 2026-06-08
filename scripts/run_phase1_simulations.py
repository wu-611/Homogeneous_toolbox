#!/usr/bin/env python3
"""
run_phase1_simulations.py — Phase 1 full-state feedback SE(3) homogeneous controller.

Simulates:
  1. Step hover: position step response
  2. Spiral trajectory: dynamic tracking
  3. Comparison: Lee geometric PD vs HPC mu=0 vs HPC mu<0

Usage: python3 scripts/run_phase1_simulations.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.quadrotor_se3 import QuadrotorSE3, exp_so3, log_so3
from simulation.simulator import Simulator
from controllers.se3_homogeneous_full import SE3HomogeneousController, LeeGeometricPD
from visualization.plotter import ResultPlotter


def run_step_response(model, controller, pos_d, yaw_d=0.0, T=8.0,
                      label='controller'):
    """Run a step response simulation."""
    sim = Simulator(model, dt=0.001, t_max=T)

    pos0 = np.zeros(3)
    vel0 = np.zeros(3)
    R0 = np.eye(3)
    omega0 = np.zeros(3)
    state0 = model.make_state(pos0, vel0, R0, omega0)
    vel_d = np.zeros(3)

    def ctrl(t, state, *args):
        return controller.compute_control(state, pos_d, vel_d, yaw_d)

    result = sim.run(ctrl, state0)
    return result


def run_trajectory_tracking(model, controller, T=10.0, label='controller'):
    """Run a spiral trajectory tracking simulation."""
    sim = Simulator(model, dt=0.001, t_max=T)

    pos0 = np.array([2.0, 0.0, -2.0])  # start on trajectory
    vel0 = np.array([0.0, 1.0, -0.1])  # match initial trajectory velocity
    R0 = np.eye(3)
    omega0 = np.zeros(3)
    state0 = model.make_state(pos0, vel0, R0, omega0)

    def ctrl(t, state, *args):
        # Spiral trajectory
        r = 2.0
        omega_traj = 0.5  # rad/s
        pos_d = np.array([r * np.cos(omega_traj * t),
                          r * np.sin(omega_traj * t),
                          -2.0 - 0.1 * t])
        vel_d = np.array([-r * omega_traj * np.sin(omega_traj * t),
                          r * omega_traj * np.cos(omega_traj * t),
                          -0.1])
        yaw_d = omega_traj * t
        return controller.compute_control(state, pos_d, vel_d, yaw_d)

    def ref_traj(t):
        r = 2.0
        omega_traj = 0.5
        pos_d = np.array([r * np.cos(omega_traj * t),
                          r * np.sin(omega_traj * t),
                          -2.0 - 0.1 * t])
        vel_d = np.array([-r * omega_traj * np.sin(omega_traj * t),
                          r * omega_traj * np.cos(omega_traj * t),
                          -0.1])
        R_d = np.eye(3)  # simplified
        omega_d = np.array([0., 0., omega_traj])
        return pos_d, vel_d, R_d, omega_d

    result = sim.run(ctrl, state0, ref_traj=ref_traj)
    return result


def main():
    print("=" * 65)
    print("Phase 1 Simulation: SE(3) Geometric Homogeneous Controller")
    print("=" * 65)

    # Model parameters (matches demo_uav)
    m = 1.4
    J = np.diag([0.0211, 0.0219, 0.0366])
    g = 9.81
    model = QuadrotorSE3(m, J, g)

    # Create controllers
    lee_pd = LeeGeometricPD(m, J, g)
    hpc_mu0 = SE3HomogeneousController(m, J, g, mu_p=0.0, mu_a=0.0)
    hpc_mu_neg = SE3HomogeneousController(m, J, g, mu_p=-0.5, mu_a=-0.5)

    controllers = {
        'Lee PD': lee_pd,
        'HPC mu=0': hpc_mu0,
        'HPC mu<0': hpc_mu_neg,
    }

    # === Scenario 1: Step hover ===
    print("\n>>> Scenario 1: Step Hover (pos_d = [1, 0.5, -2])")
    pos_d_step = np.array([1.0, 0.5, -2.0])

    results_step = {}
    for name, ctrl in controllers.items():
        print(f"  Running {name}...")
        results_step[name] = run_step_response(model, ctrl, pos_d_step, T=8.0)

    # Print final errors
    print("\n  Final position errors:")
    for name, r in results_step.items():
        pos_final = r['state'][-1, 0:3]
        err = np.linalg.norm(pos_final - pos_d_step)
        print(f"    {name:12s}: ||e_pos|| = {err:.4f} m")

    # === Scenario 2: Spiral trajectory ===
    print("\n>>> Scenario 2: Spiral Trajectory")
    results_spiral = {}
    for name, ctrl in controllers.items():
        print(f"  Running {name}...")
        results_spiral[name] = run_trajectory_tracking(model, ctrl, T=10.0)

    # Print tracking errors
    print("\n  RMS position tracking errors:")
    for name, r in results_spiral.items():
        pos = r['state'][:, 0:3]
        ref = r['ref'][:, 0:3]
        rms_err = np.sqrt(np.mean(np.sum((pos - ref)**2, axis=1)))
        print(f"    {name:12s}: RMS = {rms_err:.4f} m")

    # === Plotting ===
    print("\n>>> Generating comparison plots...")

    plotter = ResultPlotter(save_dir='./figures', use_cjk=False)

    # Step response comparison
    plotter.plot_comparison(
        results_step,
        title='Step Hover Response: Lee PD vs HPC (mu=0) vs HPC (mu<0)',
        filename='step_hover_comparison.png'
    )

    # Spiral trajectory comparison
    plotter.plot_comparison(
        results_spiral,
        title='Spiral Trajectory: Lee PD vs HPC (mu=0) vs HPC (mu<0)',
        filename='spiral_trajectory_comparison.png'
    )

    # Trajectory 3D
    plotter.plot_trajectory_3d(
        results_spiral,
        title='Spiral Trajectory Tracking Comparison',
        filename='spiral_trajectory_3d.png'
    )

    # === Summary ===
    print("\n" + "=" * 65)
    print("Simulation Summary")
    print("=" * 65)

    for name in controllers:
        r_step = results_step[name]
        r_spiral = results_spiral[name]

        pos_step = r_step['state'][:, 0:3]
        pos_d_arr = np.tile(pos_d_step, (pos_step.shape[0], 1))
        ise_step = np.trapz(np.sum((pos_step - pos_d_arr)**2, axis=1),
                            r_step['t'])

        pos_sp = r_spiral['state'][:, 0:3]
        ref_sp = r_spiral['ref'][:, 0:3]
        ise_spiral = np.trapz(np.sum((pos_sp - ref_sp)**2, axis=1),
                              r_spiral['t'])

        # Control energy
        u_step = r_step['input']
        energy_step = np.sqrt(np.trapz(np.sum(u_step**2, axis=1), r_step['t']))

        print(f"\n  {name}:")
        print(f"    Step ISE:   {ise_step:.4f} m^2*s")
        print(f"    Spiral ISE: {ise_spiral:.4f} m^2*s")
        print(f"    Step control energy: {energy_step:.2f}")

    print(f"\nFigures saved to ./figures/")
    print("Done!")


if __name__ == '__main__':
    main()
