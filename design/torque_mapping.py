"""
Torque mapping: convert virtual so(3) control to physical body torques.

Implements the inverse dynamics mapping from Zhou 2023 Eq. 23:

    M = J * (R_d' * u_hom - omega_d^x * omega + omega_d_dot) + omega x J*omega

Also handles the thrust mapping:
    F = m * (u_pos_z + g)  (thrust along body Z, gravity-compensated)
"""

import numpy as np
from models.quadrotor_se3 import hat


def map_virtual_to_torque(u_hom, J, R_d, omega, omega_d, omega_d_dot):
    """
    Convert virtual homogeneous control u_hom to physical body torque.

    Parameters
    ----------
    u_hom : (3,) ndarray
        Virtual control from attitude HPC (angular acceleration error).
    J : (3, 3) ndarray
        Inertia matrix.
    R_d : (3, 3) ndarray
        Desired rotation matrix.
    omega : (3,) ndarray
        Current body angular velocity.
    omega_d : (3,) ndarray
        Desired body angular velocity.
    omega_d_dot : (3,) ndarray
        Desired body angular acceleration.

    Returns
    -------
    M : (3,) ndarray
        Body torque command [tau_x, tau_y, tau_z].
    """
    # M = J * (R_d' * u_hom - omega_d^ x omega + omega_d_dot) + omega x J*omega
    return (J @ (R_d.T @ u_hom - np.cross(omega_d, omega) + omega_d_dot) +
            np.cross(omega, J @ omega))


def map_accel_to_thrust(u_pos_z, m, g=9.81):
    """
    Convert Z-axis acceleration command to total thrust.

    Parameters
    ----------
    u_pos_z : float
        Z-axis desired acceleration from position controller.
    m : float
        Vehicle mass.
    g : float
        Gravitational acceleration.

    Returns
    -------
    thrust : float
        Total thrust force along body Z axis.
    """
    return m * (u_pos_z + g)


def compute_control_input(u_pos, u_hom, m, g, J, R, R_d, omega, omega_d,
                          omega_d_dot):
    """
    Full control mapping: position + attitude virtual controls → [thrust, torques].

    Parameters
    ----------
    u_pos : (3,) ndarray
        Desired acceleration from position HPC [x_acc, y_acc, z_acc].
    u_hom : (3,) ndarray
        Virtual control from attitude HPC.
    m, g : float
        Mass and gravity.
    J : (3, 3) ndarray
        Inertia matrix.
    R : (3, 3) ndarray
        Current rotation matrix.
    R_d : (3, 3) ndarray
        Desired rotation matrix.
    omega : (3,) ndarray
        Current body angular velocity.
    omega_d : (3,) ndarray
        Desired body angular velocity.
    omega_d_dot : (3,) ndarray
        Desired body angular acceleration.

    Returns
    -------
    u : (4,) ndarray
        [thrust, tau_x, tau_y, tau_z].
    """
    # The Z component of u_pos is used for thrust magnitude
    # XY components are realized via attitude (body tilt)
    thrust = map_accel_to_thrust(u_pos[2], m, g)

    M = map_virtual_to_torque(u_hom, J, R_d, omega, omega_d, omega_d_dot)

    return np.array([thrust, M[0], M[1], M[2]])
