import numpy as np
import copy
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.trajectories import PiecewisePolynomial

from setup_environments import create_iiwa_plant_with_schunk_and_bin
from iiwa_block_stacking_mbp import run_sim
from quasistatic_simulator import *

#%% object positions
Kq_a = np.array([800., 600, 600, 600, 400, 200, 200, 1000, 1000])
Kq_iiwa = Kq_a[:7]
Kq_schunk = Kq_a[7:]

sphere_sdf = FindResourceOrThrow(
    'drake/examples/manipulation_station/models/sphere.sdf')

n_spheres = 2
object_sdf_paths = [sphere_sdf] * n_spheres

q_u00_list = np.zeros((n_spheres, 7))
q_u00_list[:, 0] = 1

p_start = np.array([0.45, -0.1, 0])
for i in range(n_spheres):
    q_u00_list[i, 4:] = np.random.rand(3) * 0.2 + p_start
    q_u00_list[i, -1] = (i + 1) * 0.05

q_iiwa_traj = PiecewisePolynomial.FirstOrderHold(
    [0, 50], np.zeros((2, 7)).T)
x_schunk_traj = PiecewisePolynomial.FirstOrderHold(
    [0, 50], np.array([[-0.05, 0.05, 0, 0], [-0.01, 0.01, 0, 0]]).T)
q_schunk_traj = x_schunk_traj.Block(0, 0, 2, x_schunk_traj.cols())

iiwa_log, object0_log, sim, plant, object_models = \
    run_sim(q_iiwa_traj,
            x_schunk_traj,
            Kp_iiwa=Kq_iiwa,
            Kp_schunk=Kq_schunk,
            object_sdf_paths=object_sdf_paths,
            q_u0_list=q_u00_list)


#%% Extract stationary initial condition
q_u0_list = np.zeros_like(q_u00_list)

context = sim.get_context()
diagram = sim.get_system()
context_plant = diagram.GetSubsystemContext(plant, context)

for i, object_model in enumerate(object_models):
    xi = plant.get_state_output_port(object_model).Eval(context_plant)
    q_u0_list[i] = xi[:7]
    print(i, "velocity", np.linalg.norm(xi[7:]))




#%%
q_sim = QuasistaticSimulator(
    create_iiwa_plant_with_schunk_and_bin,
    nd_per_contact=8,
    object_sdf_path=object_sdf_paths,
    joint_stiffness=Kq_a)

#%%
q0_list = [q_u0_i for q_u0_i in q_u0_list]
q0_list.append([q_iiwa_traj.value(0).squeeze(),
                q_schunk_traj.value(0).squeeze()])

q_sim.viz.vis["drake"]["contact_forces"].delete()
q_sim.UpdateConfiguration(q0_list)
q_sim.DrawCurrentConfiguration()


#%%
h = 0.05
tau_u_ext = np.array([0, 0, 0, 0., 0, -0.1])
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
    tau_u_ext_list = [tau_u_ext] * len(q_u0_list) + [None]
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