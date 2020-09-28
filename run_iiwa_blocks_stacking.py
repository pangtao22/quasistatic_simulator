import copy
import time
import numpy as np

from pydrake.math import RollPitchYaw, RotationMatrix, RigidTransform
from pydrake.trajectories import PiecewisePolynomial

from contact_aware_control.plan_runner.contact_utils import CalcIiwaQTrajectory
from contact_aware_control.plan_runner.setup_iiwa import (
    CreateIiwaControllerPlant)

from quasistatic_simulator import *
from setup_environments import (
    CreateIiwaPlantWithMultipleObjects, CreateIiwaPlantWithSchunk,
    box3d_big_sdf_path, box3d_medium_sdf_path, box3d_small_sdf_path,
    box3d_8cm_sdf_path, box3d_7cm_sdf_path)

import matplotlib.pyplot as plt

#%% Create trajectory
q_a_initial_guess = np.array([0, 0, 0, -1.75, 0, 1.0, 0])

plant_iiwa, _ = CreateIiwaControllerPlant()

durations = [1.0, 2.0, 2.0, 1.0, 1.0, 2.0]
n_blocks_to_stack = 3
l = 0.08
p_WQ_list = np.array([
    [0.55, 0, 0.10],
    [0.55, 0, 0.10],
    [0.55, 0, 0.17 + n_blocks_to_stack * l],
    [0.70, 0, 0.17 + n_blocks_to_stack * l],
    [0.70, 0, 0.17 + (n_blocks_to_stack - 1) * l],
    [0.70, 0, 0.17 + (n_blocks_to_stack - 1) * l],
    [0.55, 0, 0.25 + n_blocks_to_stack * l],
])
schunk_setpoints = [0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05]

q_a_traj_list = []
schunk_traj_list = []

for i, duration in enumerate(durations):
    q_a_traj, q_knots = CalcIiwaQTrajectory(
        plant=plant_iiwa,
        duration=duration,
        num_knot_points=10,
        p_WQ_start=p_WQ_list[i],
        p_WQ_offset=p_WQ_list[i + 1] - p_WQ_list[i],
        R_WL7_ref=RollPitchYaw(0, np.pi, 0).ToRotationMatrix(),
        q_initial_guess=q_a_initial_guess,
        p_L7Q=np.array([0, 0, 0.15]))

    q_a_traj_list.append(q_a_traj)
    schunk_traj_list.append(
        PiecewisePolynomial.FirstOrderHold(
            [0., duration],
            np.array([[-schunk_setpoints[i], schunk_setpoints[i]],
                      [-schunk_setpoints[i+1], schunk_setpoints[i+1]]]).T))

    q_a_initial_guess = q_knots[-1]


#%%
Kq_a = np.array([800., 600, 600, 600, 400, 200, 200, 400, 400])

q_sim = QuasistaticSimulator(
    CreateIiwaPlantWithSchunk,
    nd_per_contact=4,
    object_sdf_path=[box3d_8cm_sdf_path,
                     box3d_8cm_sdf_path,
                     box3d_7cm_sdf_path,
                     box3d_8cm_sdf_path],
    joint_stiffness=Kq_a)

#%%
q_u1_0 = np.array([1, 0, 0, 0, 0.55, 0, 0.04])
q_u2_0 = np.array([1, 0, 0, 0, 0.70, 0, 0.04])
q_u3_0 = np.array([1, 0, 0, 0, 0.70, 0., 0.115])
q_u4_0 = np.array([1, 0, 0, 0, 0.70, 0., 0.19])


q0_list = [q_u1_0, q_u2_0, q_u3_0, q_u4_0,
           [q_a_traj_list[0].value(0).squeeze(),
            schunk_traj_list[0].value(0).squeeze()]]

q_sim.viz.vis["drake"]["contact_forces"].delete()
q_sim.UpdateConfiguration(q0_list)
q_sim.DrawCurrentConfiguration()

#%%
h = 0.01
tau_u_ext = np.array([0, 0, 0, 0., 0, -2])
q_list = copy.deepcopy(q0_list)

q_a_log = []
q_a_cmd_log = []

input("start?")
for q_a_traj, schunk_traj in zip(q_a_traj_list, schunk_traj_list):
    n_steps = int(q_a_traj.end_time() / h)

    for i in range(n_steps):
        q_a_cmd = np.concatenate(
            [q_a_traj.value(h * i).squeeze(),
             schunk_traj.value(h * i).squeeze()])
        q_a_cmd_list = [None, None, None, None, q_a_cmd]
        tau_u_ext_list = [tau_u_ext, tau_u_ext, tau_u_ext, tau_u_ext, None]
        dq_u_list, dq_a_list = q_sim.StepAnitescu(
                q_list, q_a_cmd_list, tau_u_ext_list, h,
                is_planar=False,
                contact_detection_tolerance=0.005)

        # Update q
        q_sim.StepConfiguration(q_list, dq_u_list, dq_a_list, is_planar=False)
        q_sim.UpdateConfiguration(q_list)
        q_sim.DrawCurrentConfiguration()

        q_a_log.append(q_list[1][0].copy())
        q_a_cmd_log.append(q_a_cmd)

        # input("step?")

#%%
q_a_log = np.array(q_a_log)
q_a_cmd_log = np.array(q_a_cmd_log)

#%%
for i in range(7):
    plt.plot(q_a_log[:, i], label="q%d" % i)
    plt.plot(q_a_cmd_log[:, i], label="q_cmd")
    plt.legend()
    plt.show()
