from typing import List, Dict

import numpy as np

from pydrake.math import RollPitchYaw
from pydrake.all import (PiecewisePolynomial, PiecewiseQuaternionSlerp,
                         ModelInstanceIndex)

from iiwa_controller.iiwa_controller.utils import (
    create_iiwa_controller_plant)
from contact_aware_control.plan_runner.contact_utils import (
    CalcIiwaQTrajectory)
from setup_environments import (box3d_8cm_sdf_path, box3d_7cm_sdf_path,
                                box3d_6cm_sdf_path)


def concatenate_traj_list(traj_list: List[PiecewisePolynomial]):
    """
    Concatenates a list of PiecewisePolynomials into a single
        PiecewisePolynomial.
    :param traj_list:
    :return:
    """
    traj = traj_list[0]
    for a in traj_list[1:]:
        a.shiftRight(traj.end_time())
        traj.ConcatenateInTime(a)

    return traj


def create_initial_state_dictionary(
        q0_iiwa: np.array,
        q0_schunk: np.array,
        q_u0_list: np.array,
        model_instance_indices_u: List[ModelInstanceIndex],
        model_instance_indices_a: List[ModelInstanceIndex]
) -> Dict[ModelInstanceIndex, np.array]:
    q0_dict = dict()

    # Unactuated objects.
    for i in range(len(q_u0_list)):
        q0_dict[model_instance_indices_u[i]] = q_u0_list[i]

    # Actuated objects.
    idx_iiwa, idx_schunk = model_instance_indices_a
    q0_dict[idx_iiwa] = q0_iiwa
    q0_dict[idx_schunk] = q0_schunk

    return q0_dict


#%% Create trajectories.
q_a_initial_guess = np.array([0, 0, 0, -1.75, 0, 1.0, 0])

plant_iiwa, _ = create_iiwa_controller_plant(gravity=[0, 0, 0])

durations = np.array([1.0, 2.0, 2.0, 1.0, 1.0, 3.0]) * 2
n_blocks_to_stack = 3
l = 0.075
p_WQ_list = np.array([
    [0.555, 0, 0.10],
    [0.555, 0, 0.10],
    [0.555, 0, 0.17 + n_blocks_to_stack * l],
    [0.69, 0, 0.17 + n_blocks_to_stack * l],
    [0.69, 0, 0.17 + (n_blocks_to_stack - 1) * l + 0.005],
    [0.69, 0, 0.17 + (n_blocks_to_stack - 1) * l + 0.005],
    [0.60, 0, 0.25 + n_blocks_to_stack * l],
])
schunk_setpoints = [0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05]
# schunk_setpoints = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

q_iiwa_traj_list = []
q_schunk_traj_list = []
x_schunk_traj_list = []

num_knot_points = 10
q_iiwa_all = np.zeros((len(durations) * num_knot_points, 7))
q_schunk_all = np.zeros((len(durations) * num_knot_points, 2))

# EE orientation
R_WL7_0 = RollPitchYaw(0, np.pi, 0).ToRotationMatrix()
R_WL7_1 = RollPitchYaw(0, np.pi, np.pi/2).ToRotationMatrix()

R_WL7_traj = PiecewiseQuaternionSlerp(
    [0, durations[2]], [R_WL7_0.ToQuaternion(), R_WL7_1.ToQuaternion()])

R_WL7_list = [R_WL7_0, R_WL7_0, R_WL7_traj, R_WL7_1, R_WL7_1, R_WL7_1]

for i, duration in enumerate(durations):
    q_iiwa_traj, q_knots = CalcIiwaQTrajectory(
        plant=plant_iiwa,
        duration=duration,
        num_knot_points=num_knot_points,
        p_WQ_start=p_WQ_list[i],
        p_WQ_offset=p_WQ_list[i + 1] - p_WQ_list[i],
        R_WL7_ref=R_WL7_list[i],
        q_initial_guess=q_a_initial_guess,
        p_L7Q=np.array([0, 0, 0.15]))

    q_iiwa_traj_list.append(q_iiwa_traj)
    q_schunk_traj_list.append(
        PiecewisePolynomial.FirstOrderHold(
            [0., duration],
            np.array([[-schunk_setpoints[i], schunk_setpoints[i]],
                      [-schunk_setpoints[i+1], schunk_setpoints[i+1]]]).T))
    x_schunk_traj_list.append(
        PiecewisePolynomial.FirstOrderHold(
            [0., duration],
            np.array(
                [[-schunk_setpoints[i], schunk_setpoints[i], 0, 0],
                 [-schunk_setpoints[i+1], schunk_setpoints[i+1], 0, 0]]).T))

    q_a_initial_guess = q_knots[-1]

q_iiwa_traj = concatenate_traj_list(q_iiwa_traj_list)
q_schunk_traj = concatenate_traj_list(q_schunk_traj_list)
x_schunk_traj = concatenate_traj_list(x_schunk_traj_list)


# other constants for simulation.
Kp_iiwa = np.array([800., 600, 600, 600, 400, 200, 200])
Kp_schunk = np.array([500., 500])
Kq_a = [Kp_iiwa, Kp_schunk]

object_sdf_paths = [box3d_6cm_sdf_path,
                    box3d_8cm_sdf_path,
                    box3d_7cm_sdf_path,
                    box3d_8cm_sdf_path,
                    box3d_8cm_sdf_path,
                    box3d_7cm_sdf_path,
                    box3d_8cm_sdf_path,
                    box3d_8cm_sdf_path,
                    box3d_7cm_sdf_path,
                    box3d_8cm_sdf_path]

q_u0_list = np.zeros((10, 7))
q_u0_list[0] = [1, 0, 0, 0, 0.55, 0, 0.03]
q_u0_list[1] = [1, 0, 0, 0, 0.70, 0, 0.04]
q_u0_list[2] = [1, 0, 0, 0, 0.70, 0., 0.115]
q_u0_list[3] = [1, 0, 0, 0, 0.70, 0., 0.19]

q_u0_list[4] = [1, 0, 0, 0, 0.50, -0.2, 0.04]
q_u0_list[5] = [1, 0, 0, 0, 0.50, -0.2, 0.115]
q_u0_list[6] = [1, 0, 0, 0, 0.50, -0.2, 0.19]

q_u0_list[7] = [1, 0, 0, 0, 0.45, 0.2, 0.04]
q_u0_list[8] = [1, 0, 0, 0, 0.45, 0.2, 0.115]
q_u0_list[9] = [1, 0, 0, 0, 0.48, 0.3, 0.04]