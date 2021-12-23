import os

import tqdm
import meshcat
import numpy as np

from qsim.simulator import QuasistaticSimParameters, QuasistaticSimulator
from qsim_old.problem_definition_graze import problem_definition
from qsim.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "box_y.sdf")
model_directive_path = os.path.join(models_dir, "box_ball_graze_2d.yml")

#%%
# visualizer
viz = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")

# sim params
h = problem_definition['h']
quasistatic_sim_params = QuasistaticSimParameters(
    gravity=np.array([0, 0, 0.]),
    nd_per_contact=2,
    contact_detection_tolerance=np.inf,
    is_quasi_dynamic=True,
    mode='unconstrained',
    log_barrier_weight=1000)

# robot
Kp = problem_definition['Kq_a'].diagonal()
robot_name = 'ball'
robot_stiffness_dict = {robot_name: Kp}

# object
object_name = "box"
object_sdf_dict = {object_name: object_sdf_path}


q_sim = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=quasistatic_sim_params)

model_u = q_sim.plant.GetModelInstanceByName(object_name)
model_a = q_sim.plant.GetModelInstanceByName(robot_name)

q0_dict = {model_u: np.array([0.]),
           model_a: np.array([0, 0.])}


#%% sample dynamics
# Sample actions between the box x \in [-0.05, 0.05] and y \in [-0.05, 0.05].
n = 4000
q_a_cmd = np.random.rand(n, 2) * 0.1 - 0.05
q_next = np.zeros((n, 3))

for i in tqdm.tqdm(range(n)):
    q_sim.update_mbp_positions(q0_dict)
    q_a_cmd_dict = {model_a: q_a_cmd[i]}
    tau_ext_dict = q_sim.calc_tau_ext([])
    q_next_dict = q_sim.step(q_a_cmd_dict=q_a_cmd_dict,
                             tau_ext_dict=tau_ext_dict, h=h,
                             mode='unconstrained', gradient_mode=False)
    q_next[i] = np.hstack([q_next_dict[model_u], q_next_dict[model_a]])


#%% plot the points
# viz.delete()
n_u = problem_definition['n_u']
h = problem_definition['h']
dynamics_lcp = np.hstack([q_a_cmd, q_next[:, :n_u]])  # [x_cmd, y_cmd, x_u_next]
discontinuity_lcp = np.hstack([q_a_cmd[:, 0][:, None],
                               q_next[:, 2][:, None],
                               q_next[:, 0][:, None]])
discontinuity2_lcp = np.hstack([q_a_cmd[:, 0][:, None],
                                q_a_cmd[:, 1][:, None],
                                q_next[:, 0][:, None]])


viz["dynamics_unconstrained_100"].set_object(
    meshcat.geometry.PointCloud(
        position=dynamics_lcp.T, color=np.zeros_like(dynamics_lcp).T * 0.8))
