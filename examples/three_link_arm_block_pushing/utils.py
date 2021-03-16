from pydrake.all import RigidTransform

from examples.setup_simulation_diagram import *
from examples.setup_environments import (
    robot_sdf_path,
    create_3link_arm_controller_plant)

# Simulation parameters.
robot_name = "three_link_arm"
box_name = "box0"
X_WR = RigidTransform()
X_WR.set_translation([0, 0, 0.1])
robot_info = RobotInfo(
    sdf_path=robot_sdf_path,
    parent_model_name="WorldModelInstance",
    parent_frame_name="WorldBody",
    base_frame_name="link_0",
    X_PB=X_WR,
    joint_stiffness= np.array([1000, 1000, 1000], dtype=float))

nq_a = 3
qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [np.pi / 2, -np.pi / 2, -np.pi / 2]
qa_knots[1] = qa_knots[0] + 0.3
q_robot_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=[0, 10], samples=qa_knots.T,
    sample_dot_at_start=np.zeros(nq_a),
    sample_dot_at_end=np.zeros(nq_a))

gravity = np.array([0, 0, -10.])
h_quasistatic = 0.02
h_mbp = 1e-3


def run_comparison(box_sdf_path: str, q0_dict_str: Dict[str, np.ndarray],
                   nd_per_contact, is_visualizing=False, real_time_rate=0.0):
    #%% Quasistatic
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_a_traj_dict_str={robot_name: q_robot_traj},
        q0_dict_str=q0_dict_str,
        robot_info_dict={robot_name: robot_info},
        object_sdf_paths={box_name: box_sdf_path},
        h=h_quasistatic,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate,
        nd_per_contact=nd_per_contact)

    # %% MBP
    loggers_dict_mbp_str = run_mbp_sim(
        q_a_traj=q_robot_traj,
        q0_dict_str=q0_dict_str,
        robot_info_dict={robot_name: robot_info},
        object_sdf_paths={box_name: box_sdf_path},
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
