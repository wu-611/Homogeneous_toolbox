"""
Simulation framework for quadrotor control experiments.

Provides a generic simulation loop supporting both full-state feedback
and output-feedback controllers.
"""

import numpy as np
import time


class Simulator:
    """
    Generic simulation loop for quadrotor control.

    Supports:
    - RK4 integration of SE(3) dynamics
    - Full-state feedback and output feedback
    - Optional measurement noise
    - State and input logging
    """

    def __init__(self, model, dt=0.001, t_max=10.0):
        """
        Parameters
        ----------
        model : QuadrotorSE3
            Quadrotor dynamics model.
        dt : float
            Simulation timestep [s].
        t_max : float
            Simulation duration [s].
        """
        self.model = model
        self.dt = dt
        self.t_max = t_max
        self.N = int(t_max / dt)
        self.reset()

    def reset(self):
        """Reset logged data."""
        self.state_log = None
        self.input_log = None
        self.ref_log = None
        self.time_grid = None

    def run(self, controller, state0, ref_traj=None, noise_std=0.0):
        """
        Run simulation.

        Parameters
        ----------
        controller : callable
            Function controller(t, state) -> u (4-vector).
            For output feedback, state may be partial observation.
        state0 : (18,) ndarray
            Initial state.
        ref_traj : callable, optional
            Function ref_traj(t) -> (pos_d, vel_d, R_d, omega_d).
        noise_std : float, optional
            Standard deviation of Gaussian measurement noise on position.

        Returns
        -------
        result : dict
            Keys: 't', 'state', 'input', 'ref'
        """
        self.reset()
        state = state0.copy()

        n_state = len(state0)
        n_input = 4

        state_log = np.zeros((self.N + 1, n_state))
        input_log = np.zeros((self.N + 1, n_input))
        ref_log = np.zeros((self.N + 1, 18)) if ref_traj is not None else None
        time_grid = np.linspace(0, self.t_max, self.N + 1)

        state_log[0] = state

        for k in range(self.N):
            t = time_grid[k]

            # Measurement (with optional noise)
            if noise_std > 0:
                measured_state = state.copy()
                measured_state[0:3] += noise_std * np.random.randn(3)
            else:
                measured_state = state

            # Compute control
            if ref_traj is not None:
                pos_d, vel_d, R_d, omega_d = ref_traj(t)
                u = controller(t, measured_state, pos_d, vel_d, R_d, omega_d)
                ref_log[k] = self.model.make_state(pos_d, vel_d, R_d, omega_d)
            else:
                u = controller(t, measured_state)

            input_log[k] = u

            # Integrate
            state = self.model.step_rk4(state, u, self.dt)
            state_log[k + 1] = state

        # Final reference
        if ref_traj is not None:
            pos_d, vel_d, R_d, omega_d = ref_traj(self.t_max)
            ref_log[self.N] = self.model.make_state(pos_d, vel_d, R_d, omega_d)

        # Final control (for logging, not applied)
        input_log[self.N] = input_log[self.N - 1]

        self.state_log = state_log
        self.input_log = input_log
        self.ref_log = ref_log
        self.time_grid = time_grid

        return {
            't': time_grid,
            'state': state_log,
            'input': input_log,
            'ref': ref_log
        }

    def run_discrete_time(self, model_discrete, controller, state0, n_steps=None):
        """
        Run simulation with a discrete-time dynamics model.

        Parameters
        ----------
        model_discrete : callable
            Function model_discrete(state, u) -> next_state.
        controller : callable
            Function controller(k, state) -> u.
        state0 : ndarray
            Initial state.
        n_steps : int, optional
            Number of steps (default: N).
        """
        self.reset()
        if n_steps is None:
            n_steps = self.N

        state = state0.copy()
        n_state = len(state0)

        state_log = np.zeros((n_steps + 1, n_state))
        state_log[0] = state

        for k in range(n_steps):
            u = controller(k, state)
            state = model_discrete(state, u)
            state_log[k + 1] = state

        self.state_log = state_log
        self.time_grid = np.linspace(0, n_steps * self.dt, n_steps + 1)

        return {
            't': self.time_grid,
            'state': state_log,
        }
