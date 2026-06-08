"""
Desired attitude computation from position controller thrust direction.

Given the desired thrust direction b3_des (from position controller) and
a reference yaw angle, compute:
  - R_d  : desired rotation matrix
  - omega_d : desired angular velocity
  - omega_d_dot : desired angular acceleration

Based on Lee et al. (2010) SE(3) geometric tracking controller.
"""

import numpy as np
from models.quadrotor_se3 import hat


def compute_desired_attitude(b3_des, yaw_des=0.0):
    """
    Compute R_d from desired thrust direction and yaw reference.

    Parameters
    ----------
    b3_des : (3,) ndarray
        Desired thrust direction (unit vector in body-Z, inertial frame).
    yaw_des : float
        Desired yaw angle [rad].

    Returns
    -------
    R_d : (3, 3) ndarray
        Desired rotation matrix.
    """
    b3 = np.asarray(b3_des).flatten()
    b3 = b3 / np.linalg.norm(b3)

    # Desired body-X axis: projection of yaw reference onto plane
    # perpendicular to b3
    b1_ref = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])

    # b2_des = b3_des x b1_des / |b3_des x b1_des|
    b2 = np.cross(b3, b1_ref)
    b2_norm = np.linalg.norm(b2)
    if b2_norm < 1e-10:
        # Singularity: b3 is aligned with b1_ref
        b1_ref = np.array([0., np.cos(yaw_des), np.sin(yaw_des)])
        b2 = np.cross(b3, b1_ref)
        b2_norm = np.linalg.norm(b2)
    b2 = b2 / b2_norm

    # b1_des = b2_des x b3_des
    b1 = np.cross(b2, b3)

    R_d = np.column_stack([b1, b2, b3])
    return R_d


def compute_desired_angular_velocity(R_d, R_d_dot):
    """
    Compute desired angular velocity from R_d and its derivative.

    omega_d = (R_d' * R_d_dot)^vee

    Parameters
    ----------
    R_d : (3, 3) ndarray
        Desired rotation matrix.
    R_d_dot : (3, 3) ndarray
        Time derivative of R_d.

    Returns
    -------
    omega_d : (3,) ndarray
        Desired body angular velocity.
    """
    from models.quadrotor_se3 import vee
    return vee(R_d.T @ R_d_dot)


def compute_attitude_error(R, R_d):
    """
    Compute attitude error in exponential coordinates.

    theta_e = Log(R * R_d')

    Parameters
    ----------
    R : (3, 3) ndarray
        Current rotation matrix.
    R_d : (3, 3) ndarray
        Desired rotation matrix.

    Returns
    -------
    theta_e : (3,) ndarray
        Exponential coordinate attitude error.
    """
    from models.quadrotor_se3 import log_so3
    R_e = R @ R_d.T
    return log_so3(R_e)


def compute_omega_error(omega, omega_d, R, R_d):
    """
    Compute angular velocity error in body frame.

    omega_e = omega - R' * R_d * omega_d

    Parameters
    ----------
    omega : (3,) ndarray
        Current body angular velocity.
    omega_d : (3,) ndarray
        Desired body angular velocity.
    R : (3, 3) ndarray
        Current rotation matrix.
    R_d : (3, 3) ndarray
        Desired rotation matrix.

    Returns
    -------
    omega_e : (3,) ndarray
        Angular velocity error.
    """
    return omega - R.T @ R_d @ omega_d
