import os
import copy

import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import run_quasistatic_sim
from qsim.parser import (
    QuasistaticParser,
    QuasistaticSystemBackend,
    GradientMode,
)
from qsim.simulator import ForwardDynamicsMode
from qsim.model_paths import models_dir

# %% sim setup
q_model_path = os.path.join(models_dir, "q_sys", "allegro_hand_and_sphere.yml")

h = 0.1
duration = 2

hand_name = "allegro_hand_right"
object_name = "sphere"
nq_a = 16

qa_knots = np.zeros((2, nq_a))
qa_knots[1] += 1.0
qa_knots[1, 0] = 0
qa_knots[1, 8] = 0
qa_knots[1, 12] = 0
qa_traj = PiecewisePolynomial.FirstOrderHold([0, duration], qa_knots.T)

qu0 = np.zeros(7)
qu0[:4] = [1, 0, 0, 0]
qu0[4:] = [-0.12, 0.01, 0.07]

q_a_traj_dict_str = {hand_name: qa_traj}
q0_dict_str = {hand_name: qa_knots[0], object_name: qu0}

q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(
    is_quasi_dynamic=True,
    h=h,
    gravity=[0, 0, -10],
    log_barrier_weight=100,
    forward_mode=ForwardDynamicsMode.kLogIcecream,
)

q_sim = q_parser.make_simulator_py(internal_vis=False)
q_sim_cpp = q_parser.make_simulator_cpp()

# loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
#     q_parser=q_parser,
#     h=h,
#     backend=QuasistaticSystemBackend.CPP,
#     q_a_traj_dict_str=q_a_traj_dict_str,
#     q0_dict_str=q0_dict_str,
#     is_visualizing=True,
#     real_time_rate=1.0)

# %% look into the plant.
plant = q_sim.get_plant()
for model in q_sim.get_all_models():
    print(
        model,
        plant.GetModelInstanceName(model),
        q_sim.get_velocity_indices()[model],
    )

# %% analytical vs numerical derivatives
name_to_model_dict = q_sim.get_model_instance_name_to_index_map()
idx_a = name_to_model_dict[hand_name]
idx_u = name_to_model_dict[object_name]
q_dict = {
    idx_a: np.array(
        [
            0.03501504,
            0.75276565,
            0.74146232,
            0.83261002,
            0.63256269,
            1.02378254,
            0.64089555,
            0.82444782,
            -0.1438725,
            0.74696812,
            0.61908827,
            0.70064279,
            -0.06922541,
            0.78533142,
            0.82942863,
            0.90415436,
        ]
    ),
    idx_u: np.array(
        [
            0.96040786,
            0.07943188,
            0.26694634,
            0.00685272,
            -0.08083068,
            0.00117524,
            0.0711,
        ]
    ),
}

qa_cmd_dict = {idx_a: q_dict[idx_a] + 0.05}

# analytical gradient
sim_params = copy.deepcopy(q_sim.sim_params)
sim_params.forward_mode = ForwardDynamicsMode.kQpMp
sim_params.gradient_mode = GradientMode.kBOnly
sim_params.h = h

q_sim.update_mbp_positions(q_dict)
tau_ext_dict = q_sim.calc_tau_ext([])
q_sim.step(
    q_a_cmd_dict=qa_cmd_dict, tau_ext_dict=tau_ext_dict, sim_params=sim_params
)
dfdu_active = q_sim.get_Dq_nextDqa_cmd()

# numerical gradient
dfdu_numerical = q_sim.calc_dfdu_numerical(
    q_dict=q_dict, qa_cmd_dict=qa_cmd_dict, du=5e-4, sim_params=sim_params
)

# CPP analytic gradients
sim_params.forward_mode = ForwardDynamicsMode.kSocpMp
q_sim_cpp.update_mbp_positions(q_dict)
q_sim_cpp.step(
    q_a_cmd_dict=qa_cmd_dict, tau_ext_dict=tau_ext_dict, sim_params=sim_params
)
dfdu_active_cpp = q_sim_cpp.get_Dq_nextDqa_cmd()

#%%
# quaternion derivatives
dqdu_numerical = dfdu_numerical[-7:-3]
dqdu_analytic = dfdu_active[-7:-3]

print("dqdu numerical norm", np.linalg.norm(dqdu_numerical))
print("dqdu analytic norm", np.linalg.norm(dqdu_analytic))
print("dqdu diff norm", np.linalg.norm(dqdu_numerical - dqdu_analytic))
print("-------------------------------------------------------")

# gradient norm
diff = dfdu_numerical - dfdu_active
print("dfdu numerical norm", np.linalg.norm(dfdu_numerical))
print("dfdu analytic norm", np.linalg.norm(dfdu_active))
print("max abs diff norm", np.max(abs(diff)))
