import os

import meshcat
import numpy as np
import tqdm
from qsim.model_paths import models_dir
from qsim.simulator import (QuasistaticSimParameters, QuasistaticSimulator,
                            GradientMode, ForwardDynamicsMode)
from qsim.parser import QuasistaticParser
from qsim_old.problem_definition_graze import problem_definition

q_model_path = os.path.join(models_dir, 'q_sys', 'ball_grazing_2d.yml')

# %%
# visualizer
viz = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")

# sim params
parser = QuasistaticParser(q_model_path)
parser.set_sim_params(
    h=problem_definition['h'],
    gravity=np.array([0, 0, 0.]),
    nd_per_contact=2,
    contact_detection_tolerance=np.inf,
    is_quasi_dynamic=True,
    forward_mode=ForwardDynamicsMode.kLogPyramidMp,
    log_barrier_weight=100)
q_sim_params = QuasistaticSimulator.copy_sim_params(parser.q_sim_params)

q_sim = parser.make_simulator_py(internal_vis=False)
q_sim_cpp = parser.make_simulator_cpp()

model_u = q_sim.plant.GetModelInstanceByName("box")
model_a = q_sim.plant.GetModelInstanceByName("ball")

q0_dict = {model_u: np.array([0.]),
           model_a: np.array([0, 0.])}

n = 1000
q_a_cmd = np.random.rand(n, 2) * 0.1 - 0.05


# %% profile iterate
def run_100_times():
    for i in range(100):
        q_sim.update_mbp_positions(q0_dict)
        q_a_cmd_dict = {model_a: q_a_cmd[i]}
        tau_ext_dict = q_sim.calc_tau_ext([])
        q_next_dict = q_sim.step(q_a_cmd_dict=q_a_cmd_dict,
                                 tau_ext_dict=tau_ext_dict,
                                 sim_params=q_sim_params)

#
# cProfile.runctx('run_100_times()',
#                 globals=globals(), locals=locals(),
#                 filename='exponential_cone_qp')


# %% sample dynamics
# Sample actions between the box x \in [-0.05, 0.05] and y \in [-0.05, 0.05].
q_next = np.zeros((n, 3))
for i in tqdm.tqdm(range(n)):
    q_sim_cpp.update_mbp_positions(q0_dict)
    q_a_cmd_dict = {model_a: q_a_cmd[i]}
    tau_ext_dict = q_sim_cpp.calc_tau_ext([])
    q_sim_cpp.step(q_a_cmd_dict=q_a_cmd_dict,
                   tau_ext_dict=tau_ext_dict,
                   sim_params=q_sim_params)
    q_next_dict = q_sim_cpp.get_mbp_positions()
    q_next[i] = np.hstack([q_next_dict[model_u], q_next_dict[model_a]])

assert False
q_next2 = np.zeros((n, 3))
q_sim_params.forward_mode = ForwardDynamicsMode.kLogPyramidCvx
for i in tqdm.tqdm(range(n)):
    q_sim.update_mbp_positions(q0_dict)
    q_a_cmd_dict = {model_a: q_a_cmd[i]}
    tau_ext_dict = q_sim.calc_tau_ext([])
    q_sim.step(q_a_cmd_dict=q_a_cmd_dict,
               tau_ext_dict=tau_ext_dict,
               sim_params=q_sim_params)
    q_next = q_sim.get_mbp_positions()
    q_next2[i] = np.hstack([q_next_dict[model_u], q_next_dict[model_a]])

# %% plot the points
# viz.delete()
n_u = problem_definition['n_u']
h = problem_definition['h']
dynamics_lcp = np.hstack([q_a_cmd, q_next[:, :1]])  # [x_cmd, y_cmd, x_u_next]
discontinuity_lcp = np.hstack([q_a_cmd[:, 0][:, None],
                               q_next[:, 2][:, None],
                               q_next[:, 0][:, None]])
discontinuity2_lcp = np.hstack([q_a_cmd[:, 0][:, None],
                                q_a_cmd[:, 1][:, None],
                                q_next[:, 0][:, None]])

viz["dynamics_unconstrained_100"].set_object(
    meshcat.geometry.PointCloud(
        position=dynamics_lcp.T, color=np.zeros_like(dynamics_lcp).T * 0.8))
