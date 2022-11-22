import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import run_quasistatic_sim
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.simulator import ForwardDynamicsMode, GradientMode
from qsim.model_paths import models_dir

#%% sim setup
q_model_path = os.path.join(models_dir, "q_sys", "two_spheres_yz.yml")

h = 0.1
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h

# robot
robot_name = "sphere_yz_actuated"
object_name = "sphere_yz"

# trajectory and initial conditions.
nq_a = 2
qa_knots = np.zeros((3, nq_a))
qa_knots[0] = [0, 0.2]
qa_knots[1] = [0.8, 0.15]
qa_knots[2] = qa_knots[1]
qa_traj = PiecewisePolynomial.FirstOrderHold(
    [0, duration * 0.8, duration], qa_knots.T
)
q_a_traj_dict_str = {robot_name: qa_traj}
qu0 = np.array([0.5, 0.1])
q0_dict_str = {object_name: qu0, robot_name: qa_knots[0]}


q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(
    h=h,
    is_quasi_dynamic=True,
    forward_mode=ForwardDynamicsMode.kQpMp,
    log_barrier_weight=100,
)

# loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
#     q_parser=q_parser,
#     backend=QuasistaticSystemBackend.CPP,
#     q_a_traj_dict_str=q_a_traj_dict_str,
#     q0_dict_str=q0_dict_str,
#     is_visualizing=True,
#     real_time_rate=1.0)

q_sim = q_parser.make_simulator_cpp()
name_to_model_dict = q_sim.get_model_instance_name_to_index_map()
idx_a = name_to_model_dict[robot_name]
idx_u = name_to_model_dict[object_name]
q0_dict = {idx_a: qa_knots[0], idx_u: qu0}

# analytic gradient
sim_params = q_sim.get_sim_params()
sim_params.gradient_mode = GradientMode.kAB

#%%
q_sim.update_mbp_positions(q0_dict)
tau_ext_dict = q_sim.calc_tau_ext([])
q_sim.step(
    q_a_cmd_dict=q0_dict, tau_ext_dict=tau_ext_dict, sim_params=sim_params
)

A = q_sim.get_Dq_nextDq()
B = q_sim.get_Dq_nextDqa_cmd()
