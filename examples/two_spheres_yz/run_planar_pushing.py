import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import run_quasistatic_sim
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.simulator import ForwardDynamicsMode, GradientMode
from qsim.model_paths import models_dir
from qsim_cpp import FiniteDiffGradientCalculator

# %% sim setup
q_model_path = os.path.join(models_dir, "q_sys", "two_spheres_yz.yml")

h = 0.1
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h

# robot
robot_name = "sphere_yz_actuated"
object_name = "sphere_yz"


q_a0 = np.zeros(2)
q_u0 = np.array([0.3, 0.1])

q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(
    h=h,
    is_quasi_dynamic=True,
    forward_mode=ForwardDynamicsMode.kQpMp,
    log_barrier_weight=100,
)

q_sim = q_parser.make_simulator_cpp()
q_sim_py = q_parser.make_simulator_py(internal_vis=True)

name_to_model_dict = q_sim.get_model_instance_name_to_index_map()
idx_a = name_to_model_dict[robot_name]
idx_u = name_to_model_dict[object_name]

# %%
q0_dict = {idx_a: q_a0, idx_u: q_u0}
q_a_cmd_dict = {idx_a: np.array([0.1, 0])}

# analytic gradient
sim_params = q_sim.get_sim_params()
sim_params.gradient_mode = GradientMode.kAB
sim_params.forward_mode = ForwardDynamicsMode.kSocpMp

# %%
q_sim.update_mbp_positions(q0_dict)
tau_ext_dict = q_sim.calc_tau_ext([])
q_sim.step(
    q_a_cmd_dict=q_a_cmd_dict, tau_ext_dict=tau_ext_dict, sim_params=sim_params
)
q_next = q_sim.get_q_vec_from_dict(q_sim.get_mbp_positions())
A = q_sim.get_Dq_nextDq()
B = q_sim.get_Dq_nextDqa_cmd()

# %%
fd_gradient_calculator = FiniteDiffGradientCalculator(q_sim)
q_nominal = q_sim.get_q_vec_from_dict(q0_dict)
u_nominal = q_sim.get_q_a_cmd_vec_from_dict(q_a_cmd_dict)
A_numerical = fd_gradient_calculator.calc_A(
    q_nominal, u_nominal, 1e-3, sim_params
)
B_numerical = fd_gradient_calculator.calc_B(
    q_nominal, u_nominal, 1e-3, sim_params
)
