import os
import cProfile
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


# %% sample dynamics
# Sample actions between the box x \in [-0.05, 0.05] and y \in [-0.05, 0.05].
q_sim_params.gradient_mode = GradientMode.kNone


def calc_dynamics(forward_mode: ForwardDynamicsMode):
    q_next = np.zeros((n, 3))
    q_sim_params.forward_mode = forward_mode
    for i in tqdm.tqdm(range(n)):
        q_a_cmd_dict = {model_a: q_a_cmd[i]}
        tau_ext_dict = q_sim_cpp.calc_tau_ext([])
        q_sim_cpp.update_mbp_positions(q0_dict)
        q_sim_cpp.step(q_a_cmd_dict=q_a_cmd_dict,
                       tau_ext_dict=tau_ext_dict,
                       sim_params=q_sim_params)
        q_next_dict = q_sim_cpp.get_mbp_positions()
        q_next[i] = np.hstack([q_next_dict[model_a], q_next_dict[model_u]])

    return q_next

q_next_pyramid = calc_dynamics(ForwardDynamicsMode.kLogPyramidMy)
q_next_icecream = calc_dynamics(ForwardDynamicsMode.kLogIcecream)
q_next_qp = calc_dynamics(ForwardDynamicsMode.kQpMp)


#%%
i = 30
q_a_cmd_dict = {model_a: q_a_cmd[i]}
tau_ext_dict = q_sim_cpp.calc_tau_ext([])




# %% plot the points
viz.delete()
n_u = problem_definition['n_u']
h = problem_definition['h']
# [x_cmd, y_cmd, x_u_next]
dynamics_pyramid = np.hstack([q_a_cmd, q_next_pyramid[:, -1:]])
dynamics_icecream = np.hstack([q_a_cmd, q_next_icecream[:, -1:]])
dynamics_exact = np.hstack([q_a_cmd, q_next_qp[:, -1:]])


viz["dynamics_pyramid"].set_object(
    meshcat.geometry.PointCloud(
        position=dynamics_pyramid.T,
        color=np.zeros_like(dynamics_pyramid).T * 0.8))

viz["dynamics_icecream"].set_object(
    meshcat.geometry.PointCloud(
        position=dynamics_icecream.T,
        color=np.ones_like(dynamics_icecream).T * 0.8))


color_exact = np.ones_like(dynamics_exact)
color_exact[:] = [1, 0, 0]
viz["dynamics_exact"].set_object(
    meshcat.geometry.PointCloud(
        position=dynamics_exact.T,
        color=color_exact.T))

#%%

