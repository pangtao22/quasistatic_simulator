import os.path

from examples.setup_simulation_diagram import *
from examples.setup_environments import (model_dir_path,
    create_3link_arm_controller_plant)

# Simulation parameters.
gravity = np.array([0, 0, -10.])
robot_name = "arm"
box_name = "box0"
robot_stiffness_dict = {robot_name: np.array([1000, 1000, 1000], dtype=float)}
h_quasistatic = 0.02
h_mbp = 1e-3

# Robot joint trajectory.
nq_a = 3
qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [np.pi / 2, -np.pi / 2, -np.pi / 2]
qa_knots[1] = qa_knots[0] + 0.3
q_robot_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=[0, 10], samples=qa_knots.T,
    sample_dot_at_start=np.zeros(nq_a),
    sample_dot_at_end=np.zeros(nq_a))

# model directive paths
model_directive_path = os.path.join(
    model_dir_path, 'three_link_arm_and_ground.yml')


def run_comparison(box_sdf_path: str, q0_dict_str: Dict[str, np.ndarray],
                   quasistatic_sim_params: QuasistaticSimParameters,
                   is_visualizing=False, real_time_rate=0.0):
    #%% Quasistatic
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        model_directive_path=model_directive_path,
        object_sdf_paths={box_name: box_sdf_path},
        q_a_traj_dict_str={robot_name: q_robot_traj}, q0_dict_str=q0_dict_str,
        robot_stiffness_dict=robot_stiffness_dict, h=h_quasistatic,
        sim_params=quasistatic_sim_params,
        is_visualizing=is_visualizing, real_time_rate=real_time_rate)

    # %% MBP
    loggers_dict_mbp_str = run_mbp_sim(
        model_directive_path=model_directive_path,
        object_sdf_paths={box_name: box_sdf_path},
        q_a_traj=q_robot_traj,
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict=robot_stiffness_dict,
        create_controller_plant=create_3link_arm_controller_plant,
        h=h_mbp,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate)

    # Extracting iiwa configuration logs.
    n_qu = q0_dict_str[box_name].size
    q_box_log_mbp = loggers_dict_mbp_str[box_name].data()[:n_qu].T
    q_robot_log_mbp = loggers_dict_mbp_str[robot_name].data()[:nq_a].T
    t_mbp = loggers_dict_mbp_str[robot_name].sample_times()

    q_robot_log_quasistatic = loggers_dict_quasistatic_str[robot_name].data().T
    q_box_log_quasistatic = loggers_dict_quasistatic_str[box_name].data().T
    t_quasistatic = loggers_dict_quasistatic_str[robot_name].sample_times()

    return (q_robot_log_mbp, q_box_log_mbp, t_mbp,
            q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic,
            q_sys)
