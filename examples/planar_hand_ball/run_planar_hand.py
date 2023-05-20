import os

import numpy as np
from examples.setup_simulations import run_quasistatic_sim
from pydrake.all import PiecewisePolynomial
from qsim.model_paths import models_dir
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.simulator import GradientMode
from qsim_cpp import ForwardDynamicsMode

# %% sim setup
q_model_path = os.path.join(models_dir, "q_sys", "planar_hand_ball.yml")

h = 0.1
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h

# model instance names.
robot_l_name = "arm_left"
robot_r_name = "arm_right"
object_name = "sphere"

# trajectory and initial conditions.
nq_a = 2
qa_l_knots = np.zeros((2, nq_a))
qa_l_knots[0] = [-np.pi / 4, -np.pi / 4]
q_robot_l_traj = PiecewisePolynomial.ZeroOrderHold([0, T * h], qa_l_knots.T)

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 4]
q_robot_r_traj = PiecewisePolynomial.ZeroOrderHold([0, T * h], qa_r_knots.T)

q_a_traj_dict_str = {robot_l_name: q_robot_l_traj, robot_r_name: q_robot_r_traj}

q_u0 = np.array([0, 0.5, 0])

q0_dict_str = {
    object_name: q_u0,
    robot_l_name: qa_l_knots[0],
    robot_r_name: qa_r_knots[0],
}

# %% run sim.
if __name__ == "__main__":
    q_parser = QuasistaticParser(q_model_path)
    q_parser.set_sim_params(
        h=h,
        is_quasi_dynamic=True,
        forward_mode=ForwardDynamicsMode.kLogPyramidCvx,
        log_barrier_weight=100,
        gravity=np.array([0, 0, -10.0]),
    )

    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_parser=q_parser,
        backend=QuasistaticSystemBackend.PYTHON,
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        is_visualizing=True,
        real_time_rate=1.0,
    )

    # %% look into the plant.
    plant = q_sys.plant
    for model in q_sys.q_sim.models_all:
        print(
            model,
            plant.GetModelInstanceName(model),
            q_sys.q_sim.velocity_indices[model],
        )

    # %% derivatives.
    q_sim = q_sys.q_sim
    name_to_model_dict = q_sim.get_model_instance_name_to_index_map()
    idx_l = name_to_model_dict[robot_l_name]
    idx_r = name_to_model_dict[robot_r_name]
    idx_o = name_to_model_dict[object_name]
    q_dict = {
        idx_o: [0, 0.316, 0],
        idx_l: [-0.775, -0.785],
        idx_r: [0.775, 0.785],
    }

    # analytical gradient
    sim_params = q_sim.get_sim_parmas_copy()
    sim_params.h = 0.05
    sim_params.forward_mode = ForwardDynamicsMode.kQpMp
    sim_params.gradient_mode = GradientMode.kBOnly

    q_sim.update_mbp_positions(q_dict)
    tau_ext_dict = q_sim.calc_tau_ext([])
    q_sim.step(q_a_cmd_dict=q_dict, tau_ext_dict=tau_ext_dict, sim_params=sim_params)
    dfdu_active = q_sim.get_Dq_nextDqa_cmd()

    # numerical gradient
    dfdu_numerical = q_sim.calc_dfdu_numerical(
        q_dict=q_dict, qa_cmd_dict=q_dict, du=1e-4, sim_params=sim_params
    )

    print("dfdu_error", np.linalg.norm(dfdu_numerical - dfdu_active))
