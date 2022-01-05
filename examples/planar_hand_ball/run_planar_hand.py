import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import (
    run_quasistatic_sim)
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.model_paths import models_dir
from qsim.simulator import GradientMode


#%% sim setup
q_model_path = os.path.join(models_dir, 'q_sys', 'planar_hand_ball.yml')

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
q_robot_l_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_l_knots.T)

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 4]
q_robot_r_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_r_knots.T)

q_a_traj_dict_str = {robot_l_name: q_robot_l_traj,
                     robot_r_name: q_robot_r_traj}

q_u0 = np.array([0, 0.5, 0])

q0_dict_str = {object_name: q_u0,
               robot_l_name: qa_l_knots[0],
               robot_r_name: qa_r_knots[0]}


#%% run sim.
if __name__ == "__main__":
    q_parser = QuasistaticParser(q_model_path)
    q_parser.set_sim_params(is_quasi_dynamic=True, gravity=np.array([0, 0, -10.]))

    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_parser=q_parser,
        h=h,
        backend=QuasistaticSystemBackend.PYTHON,
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        is_visualizing=True,
        real_time_rate=1.0)

#%% look into the plant.
    plant = q_sys.plant
    for model in q_sys.q_sim.models_all:
        print(model, plant.GetModelInstanceName(model),
              q_sys.q_sim.velocity_indices[model])

#%% derivatives.
    q_sim = q_sys.q_sim
    name_to_model_dict = q_sim.get_model_instance_name_to_index_map()
    idx_l = name_to_model_dict[robot_l_name]
    idx_r = name_to_model_dict[robot_r_name]
    idx_o = name_to_model_dict[object_name]
    q_dict = {idx_o: [0, 0.316, 0],
              idx_l: [-0.775, -0.785],
              idx_r: [0.775, 0.785]}

    # numerical gradient
    dfdu_numerical = q_sim.calc_dfdu_numerical(
        q_dict=q_dict, qa_cmd_dict=q_dict, du=1e-3, h=h)

    # analytical gradient
    q_sim.update_mbp_positions(q_dict)
    tau_ext_dict = q_sim.calc_tau_ext([])
    q_sim.step(q_a_cmd_dict=q_dict, tau_ext_dict=tau_ext_dict, h=h,
               mode="qp_mp", gradient_mode=GradientMode.kBOnly,
               grad_from_active_constraints=True)
    dfdu_active = q_sim.get_Dq_nextDqa_cmd()


#%% index for tau_a.
    indices = []
    for model in q_sys.q_sim.models_actuated:
        indices += q_sys.q_sim.velocity_indices[model].tolist()
    indices.sort()
    indices_map = {j: i for i, j in enumerate(indices)}

#%% construct q and v vectors of MBP from log.
    logger_qu = loggers_dict_quasistatic_str[object_name]
    q_log = np.zeros((T, plant.num_positions()))
    v_log = np.zeros((T, plant.num_velocities()))
    tau_a_log = np.zeros((T - 1, plant.num_actuated_dofs()))

    for name, logger in loggers_dict_quasistatic_str.items():
        model = name_to_model_dict[name]
        for i, j in enumerate(q_sys.q_sim.velocity_indices[model]):
            q_log[:, j] = logger.data().T[:, i]

    v_log[1:, :] = (q_log[1:, :] - q_log[:-1, :]) / h

    for name in robot_stiffness_dict.keys():
        model = name_to_model_dict[name]
        logger_qa = loggers_dict_quasistatic_str[name]
        idx_v = q_sys.q_sim.velocity_indices[model]
        idx_tau_a = [indices_map[i] for i in idx_v]
        for l in range(T - 1):
            qa_l = logger_qa.data().T[l]
            qa_l1_cmd = q_a_traj_dict_str[name].value((l + 1) * h).squeeze()
            tau_a_log[l][idx_tau_a] = Kp * (qa_l1_cmd - qa_l)

