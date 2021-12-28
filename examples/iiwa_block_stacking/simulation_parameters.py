import os
from typing import List

import numpy as np

from pydrake.math import RollPitchYaw
from pydrake.all import (PiecewisePolynomial, RigidTransform)
from qsim.simulator import (
    QuasistaticSimParameters)
from robotics_utilities.iiwa_controller.utils import (
    create_iiwa_controller_plant)
from qsim.model_paths import (models_dir, box3d_8cm_sdf_path,
                              box3d_7cm_sdf_path, box3d_6cm_sdf_path)

from .inverse_kinematics import calc_iwa_trajectory_for_point_tracking


q_model_path = os.path.join(models_dir, 'q_sys', 'iiwa_and_boxes.yml')


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


#%% Create trajectories.
q_a_initial_guess = np.array([0, 0, 0, -1.75, 0, 1.0, 0])
plant_iiwa, _ = create_iiwa_controller_plant(gravity=[0, 0, 0])

durations = np.array([1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0]) * 2
n_blocks_to_stack = 3
l = 0.075
p_WQ_list = np.array([
    [0.555, 0, 0.10],
    [0.555, 0, 0.10],
    [0.555, 0, 0.17 + n_blocks_to_stack * l],
    [0.69, 0, 0.17 + n_blocks_to_stack * l],
    [0.69, 0, 0.17 + (n_blocks_to_stack - 1) * l + 0.005],
    [0.69, 0, 0.17 + (n_blocks_to_stack - 1) * l + 0.005],
    [0.69, 0, 0.25 + n_blocks_to_stack * l],
    [0.555, 0, 0.25 + n_blocks_to_stack * l],
])
schunk_setpoints = [0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]

# frame L7 orientation
R_WL7_0 = RollPitchYaw(0, np.pi, 0).ToRotationMatrix()
R_WL7_1 = RollPitchYaw(0, np.pi, np.pi/2).ToRotationMatrix()
R_WL7_list = [
    R_WL7_0, R_WL7_0, R_WL7_0, R_WL7_1, R_WL7_1, R_WL7_1, R_WL7_1, R_WL7_0]


q_iiwa_traj_list = []
q_schunk_traj_list = []
x_schunk_traj_list = []

for i, duration in enumerate(durations):
    q_iiwa_traj, q_knots = calc_iwa_trajectory_for_point_tracking(
        plant=plant_iiwa,
        duration=duration,
        num_knot_points=10,
        p_WQ_start=p_WQ_list[i],
        p_WQ_offset=p_WQ_list[i + 1] - p_WQ_list[i],
        R_WL7_start=R_WL7_list[i],
        R_WL7_final=R_WL7_list[i + 1],
        q_initial_guess=q_a_initial_guess,
        p_L7Q=np.array([0, 0, 0.15]))

    q_iiwa_traj_list.append(q_iiwa_traj)
    q_schunk_traj_list.append(
        PiecewisePolynomial.FirstOrderHold(
            [0., duration],
            np.array([[-schunk_setpoints[i], schunk_setpoints[i]],
                      [-schunk_setpoints[i+1], schunk_setpoints[i+1]]]).T))

    q_a_initial_guess = q_knots[-1]

q_iiwa_traj = concatenate_traj_list(q_iiwa_traj_list)
q_schunk_traj = concatenate_traj_list(q_schunk_traj_list)

# other constants for simulation.
iiwa_name = "iiwa"
schunk_name = "schunk"

X_L7E = RigidTransform(
    RollPitchYaw(np.pi/2, 0, np.pi/2), np.array([0, 0, 0.114]))

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

q0_dict_str = {"box%i" % i: q_u0_i for i, q_u0_i in enumerate(q_u0_list)}
t_start = q_iiwa_traj.start_time()
q0_dict_str[iiwa_name] = q_iiwa_traj.value(t_start).ravel()
q0_dict_str[schunk_name] = q_schunk_traj.value(t_start).ravel()

q_a_traj_dict_str = {iiwa_name: q_iiwa_traj, schunk_name: q_schunk_traj}
