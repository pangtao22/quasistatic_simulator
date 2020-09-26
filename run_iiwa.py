import copy
import time
import numpy as np

from pydrake.math import RollPitchYaw, RotationMatrix, RigidTransform

from contact_aware_control.plan_runner.contact_utils import CalcIiwaQTrajectory
from contact_aware_control.plan_runner.setup_iiwa import (
    CreateIiwaControllerPlant)

from quasistatic_simulator import *
from setup_environments import CreatePlantForIiwaWithMultipleObjects
from sim_params_3link_arm import (
    box3d_big_sdf_path, box3d_medium_sdf_path, box3d_small_sdf_path)

#%% Create trajectory
q_a_initial_guess = np.array([0, 0, 0, -1.75, 0, 1.0, 0])

plant_iiwa, _ = CreateIiwaControllerPlant()
q_a_traj, q_knots = CalcIiwaQTrajectory(
    plant=plant_iiwa,
    duration=5.0,
    num_knot_points=10,
    p_WQ_start=np.array([0.5, 0, 0.4]),
    p_WQ_offset=np.array([0.3, 0, 0.0]),
    R_WL7_ref=RollPitchYaw(0, np.pi / 4 * 3, 0).ToRotationMatrix(),
    q_initial_guess=q_a_initial_guess,
    p_L7Q=np.array([0, 0, 0.15]))


#%%
Kq_a = np.array([800., 600, 600, 600, 400, 200, 200])

q_sim = QuasistaticSimulator(
    CreatePlantForIiwaWithMultipleObjects,
    nd_per_contact=4,
    object_sdf_path=[box3d_medium_sdf_path, box3d_small_sdf_path],
    joint_stiffness=Kq_a)

#%%
q_u1_0 = np.array([1, 0, 0, 0, 0.85, 0, 0.3])
q_u2_0 = np.array([1, 0, 0, 0, 1.5, 0.4, 0.25])
q0_list = [q_u1_0, q_u2_0, q_a_traj.value(0).squeeze()]

q_sim.viz.vis["drake"]["contact_forces"].delete()
q_sim.UpdateConfiguration(q0_list)
q_sim.DrawCurrentConfiguration()

#%%
h = 0.01
tau_u_ext = np.array([0, 0, 0, 0., 0, -10])
n_steps = int(q_a_traj.end_time() / h)
q_list = copy.deepcopy(q0_list)

input("start?")
for i in range(n_steps):
    q_a_cmd = q_a_traj.value(h * i).squeeze()
    q_a_cmd_list = [None, None, q_a_cmd]
    tau_u_ext_list = [tau_u_ext, tau_u_ext, None]
    # q_a_cmd_list = [q_a_cmd]
    # tau_u_ext_list = []
    dq_u_list, dq_a_list = q_sim.StepAnitescu(
            q_list, q_a_cmd_list, tau_u_ext_list, h,
            is_planar=False,
            contact_detection_tolerance=0.01)

    # Update q
    q_sim.StepConfiguration(q_list, dq_u_list, dq_a_list, is_planar=False)
    q_sim.UpdateConfiguration(q_list)
    q_sim.DrawCurrentConfiguration()

    input("step?")


