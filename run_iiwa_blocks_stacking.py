import copy
import time
import numpy as np

from pydrake.math import RollPitchYaw, RotationMatrix, RigidTransform
from pydrake.common.eigen_geometry import Quaternion
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from contact_aware_control.plan_runner.contact_utils import CalcIiwaQTrajectory
from iiwa_controller.iiwa_controller.utils import (
    create_iiwa_controller_plant)

from quasistatic_simulator import *
from setup_environments import (
    CreateIiwaPlantWithMultipleObjects, create_iiwa_plant_with_schunk,
    box3d_big_sdf_path, box3d_medium_sdf_path, box3d_small_sdf_path,
    box3d_8cm_sdf_path, box3d_7cm_sdf_path, box3d_6cm_sdf_path)
from iiwa_block_stacking_mbp import run_sim

import matplotlib.pyplot as plt

#%% Create trajectory
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


def concatenate_traj_list(traj_list: List[PiecewisePolynomial]):
    traj = traj_list[0]
    for a in traj_list[1:]:
        a.shiftRight(traj.end_time())
        traj.ConcatenateInTime(a)

    return traj

q_iiwa_traj = concatenate_traj_list(q_iiwa_traj_list)
q_schunk_traj = concatenate_traj_list(q_schunk_traj_list)
x_schunk_traj = concatenate_traj_list(x_schunk_traj_list)


#%%
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

Kq_a = np.array([800., 600, 600, 600, 400, 200, 200, 500, 500])
Kq_iiwa = Kq_a[:7]
Kq_schunk = Kq_a[7:]
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

#%%
q_sim = QuasistaticSimulator(
    create_iiwa_plant_with_schunk,
    nd_per_contact=4,
    object_sdf_paths=object_sdf_paths,
    joint_stiffness=Kq_a)

#%%
q0_list = [q_u0_i for q_u0_i in q_u0_list]
q0_list.append([q_iiwa_traj_list[0].value(0).squeeze(),
                q_schunk_traj_list[0].value(0).squeeze()])

q_sim.viz.vis["drake"]["contact_forces"].delete()
q_sim.UpdateConfiguration(q0_list)
q_sim.DrawCurrentConfiguration()

#%%
h = 0.2
tau_u0_ext = np.array([0, 0, 0, 0., 0, -5])
tau_u_other_ext = np.array([0, 0, 0, 0., 0, -50])

q_list = copy.deepcopy(q0_list)

q_a_log = []
q_log = []
q_a_cmd_log = []

input("start?")
n_steps = int(q_iiwa_traj.end_time() / h)

for i in range(n_steps):
    q_a_cmd = np.concatenate(
        [q_iiwa_traj.value(h * i).squeeze(),
         q_schunk_traj.value(h * i).squeeze()])
    q_a_cmd_list = [None] * len(q_u0_list) + [q_a_cmd]
    tau_u_ext_list = \
        [tau_u0_ext] + [tau_u_other_ext] * (len(q_u0_list)-1) + [None]
    dq_u_list, dq_a_list = q_sim.StepAnitescu(
            q_list, q_a_cmd_list, tau_u_ext_list, h,
            is_planar=False,
            contact_detection_tolerance=0.005)

    # Update q
    q_sim.StepConfiguration(q_list, dq_u_list, dq_a_list, is_planar=False)
    q_sim.UpdateConfiguration(q_list)
    q_sim.DrawCurrentConfiguration()

    q_a_log.append(np.concatenate(q_list[-1]))
    q_a_cmd_log.append(q_a_cmd)
    q_log.append(copy.deepcopy(q_list))

    # time.sleep(h)
    # print("t = ", i * h)
    # input("step?")


#%%
def extract_log_for_object(i: int):
    n = len(q_log)
    m = len(q_log[0][i])
    q_i_log = np.zeros((n, m))
    for t, q_t in enumerate(q_log):
        q_i_log[t] = q_t[i]
    return q_i_log

q_a_log = np.array(q_a_log)
q_a_cmd_log = np.array(q_a_cmd_log)
q_u0_log = extract_log_for_object(0)

#
# for i in range(len(q_log[0])):
#     print(i, q_log[0][i][-3:], q_log[-1][i][-3:])

error_qa = np.zeros_like(q_a_log[:, :7])
t_s = np.arange(n_steps) * h
for i, t in enumerate(t_s):
    error_qa[i] = q_iiwa_traj.value(t).squeeze() - q_a_log[i, :7]

np.save("qa_10cube_error_h{}".format(h), error_qa)
np.save("qa_10cube_q_h{}".format(h), q_a_log)  # IIWA only
np.save("cube0_10cube_q_h{}".format(h), q_u0_log)

#%% log playback
stride = 500
for i in range(0, len(q_log), stride):
    q_sim.UpdateConfiguration(q_log[i])
    q_sim.DrawCurrentConfiguration()
    time.sleep(h * stride)


#%%
h_mbp = 0.001
# h_mbp = 0.001088
iiwa_log, schunk_log, object0_log = run_sim(q_iiwa_traj, x_schunk_traj,
                                Kp_iiwa=Kq_iiwa,
                                Kp_schunk=Kq_schunk,
                                object_sdf_paths=object_sdf_paths,
                                q_u0_list=q_u0_list,
                                time_step=h_mbp)

#  save ground truth logs
na = 7
t_s = iiwa_log.sample_times()
q_iiwa_log = iiwa_log.data()[:na].T
q_schunk_log = schunk_log.data()[:2].T
q_a_log = np.hstack((q_iiwa_log, q_schunk_log))

# error_qa = np.zeros_like(q_a_log)
#
# for i, t in enumerate(t_s):
#     error_qa[i] = q_iiwa_traj.value(t).squeeze() - q_iiwa_log[i]

# np.save("qa_10cube_error_mbp_h{}".format(h_mbp), error_qa)
np.save("qa_10cube_q_mbp_h{}".format(h_mbp), q_a_log)
np.save("cube0_10cube_q_mbp_h{}".format(h_mbp), object0_log.data()[:7].T)
