#%%
from typing import List, Union

import numpy as np

from pydrake.all import PiecewiseQuaternionSlerp, PiecewisePolynomial
from pydrake.common.eigen_geometry import Quaternion, AngleAxis


def calc_error_integral(q_knots, t, q_gt_traj):

    n = len(t)
    assert len(q_knots) == len(t)
    e_vec = np.zeros(n - 1)
    t_e = np.zeros(n - 1)
    e = 0.

    for i in range(n-1):
        t_i = (t[i] + t[i + 1]) / 2
        q_i = (q_knots[i] + q_knots[i + 1]) / 2
        dt = t[i + 1] - t[i]
        dq = q_i - q_gt_traj.value(t_i).ravel()
        e_vec[i] = np.linalg.norm(dq)
        t_e[i] = t_i
        e += e_vec[i] * dt

    return e, e_vec, t_e


def get_angle_from_quaternion(q: np.array):
    q /= np.linalg.norm(q)
    a = AngleAxis(Quaternion(q))
    return a.angle()


def convert_quaternion_array_to_eigen_quaternion_traj(q_array: np.array,
                                                      t: np.array):
    """
    :param q_array: (n, 4) array where q_array[i] is a quaternion.
    :param t: (n,) array of times.
    :return:
    """
    quaternion_list = [Quaternion(q / np.linalg.norm(q)) for q in q_array]
    return PiecewiseQuaternionSlerp(t, quaternion_list)


def calc_quaternion_error_integral(
        q_list: Union[List[Quaternion], np.array],
        t: np.array,
        q_traj: PiecewiseQuaternionSlerp):
    assert q_traj.is_time_in_range(t[0]) and q_traj.is_time_in_range(t[-1])
    assert len(q_list) == len(t)

    angle_diff_list = []

    for q_i, t_i in zip(q_list, t):
        q1 = Quaternion(q_i / np.linalg.norm(q_i))
        q2 = Quaternion(q_traj.value(t_i))
        angle_diff_list.append(AngleAxis(q1.inverse().multiply(q2)).angle())

    angle_diff_list = np.array(angle_diff_list)

    zero_traj = PiecewisePolynomial.ZeroOrderHold(
        [t[0], t[-1]], np.array([[0, 0.]]))
    return calc_error_integral(angle_diff_list, t, zero_traj)


def calc_pose_error_integral(pose_list_1, pose_list_2, t):
    """
    :param pose_list is a 2D numpy array of shape (n, 7).
        pose_list[i] is a (7,) array representing the pose of a rigid body at
        time step i.  pose_list[i, :4] is a quaternion and
        pose_list[i, 4:] is a point in R^3.
    :return:
    """
    q_diff_list = calc_quaternion_difference(
        pose_list_1[:, :4], pose_list_2[:, :4])

    angles_diff = np.array([AngleAxis(q).angle() for q in q_diff_list])
    xyz_diff = pose_list_1[:, 4:] - pose_list_2[:, 4:]
    e_angles, e_angles_vec, t_e = calc_error_integral()
