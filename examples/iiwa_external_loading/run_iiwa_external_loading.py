from matplotlib import pyplot as plt
from pydrake.all import RigidTransform

from examples.setup_simulation_diagram import *
from iiwa_controller.iiwa_controller.utils import create_iiwa_controller_plant
from examples.setup_environments import iiwa_sdf_path_drake


#%% simulation parameters.
Kp_iiwa = np.array([800., 600, 600, 600, 400, 200, 200])
robot_name = "iiwa7"
robot_info = RobotInfo(
    sdf_path=iiwa_sdf_path_drake,
    parent_model_name="WorldModelInstance",
    parent_frame_name="WorldBody",
    base_frame_name="iiwa_link_0",
    X_PB=RigidTransform(),
    joint_stiffness=Kp_iiwa)

nq_a = 7
q_iiwa_knots = np.zeros((2, 7))
q_iiwa_knots[0] = [0, 0, 0, -1.70, 0, 1.0, 0]
q_iiwa_knots[1] = q_iiwa_knots[0]
q_iiwa_traj = PiecewisePolynomial.FirstOrderHold([0, 2], q_iiwa_knots.T)

q0_dict_str = {robot_name: q_iiwa_knots[0]}
gravity = np.array([0, 0, -10.])

F_WB = np.zeros((2, 3))
F_WB[1] = [0, 0, -100.]
F_WB_traj = PiecewisePolynomial.FirstOrderHold(
    [0, q_iiwa_traj.end_time() / 2], F_WB.T)


def run_comparison(is_visualizing: bool, real_time_rate: float):
    #%% Quasistatic.
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_a_traj_dict_str={robot_name: q_iiwa_traj},
        q0_dict_str=q0_dict_str,
        robot_info_dict={robot_name: robot_info},
        object_sdf_paths=dict(),
        h=0.2,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate,
        body_name="iiwa_link_7",
        F_WB_traj=F_WB_traj)

    #%% MBP.
    loggers_dict_mbp_str = run_mbp_sim(
        q_a_traj=q_iiwa_traj,
        q0_dict_str=q0_dict_str,
        robot_info_dict={robot_name: robot_info},
        object_sdf_paths=dict(),
        create_controller_plant=create_iiwa_controller_plant,
        h=1e-4,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate,
        body_name="iiwa_link_7",
        F_WB_traj=F_WB_traj)

    return loggers_dict_quasistatic_str, loggers_dict_mbp_str


if __name__ == "__main__":
    loggers_dict_quasistatic_str, loggers_dict_mbp_str = run_comparison(
        is_visualizing=True, real_time_rate=1.0)

    #%%
    nq = 7
    iiwa_log_mbp = loggers_dict_mbp_str[robot_name]
    q_iiwa_mbp = iiwa_log_mbp.data().T[:, :nq]
    t_mbp = iiwa_log_mbp.sample_times()

    iiwa_log_qs = loggers_dict_quasistatic_str[robot_name]
    q_iiwa_qs = iiwa_log_qs.data().T[:, :nq]
    t_qs = iiwa_log_qs.sample_times()

    #%% plot iiwa joint angles.
    fig, axes = plt.subplots(nq, 1, figsize=(4, 10), dpi=150)
    for i in range(nq):
        axes[i].plot(t_mbp, q_iiwa_mbp[:, i], label="mbp")
        axes[i].plot(t_qs, q_iiwa_qs[:, i], label="quasistatic")
        axes[i].set_ylabel("joint {} [rad]".format(i + 1))

    axes[-1].set_xlabel("t [s]")
    plt.legend()
    plt.show()
