import os
from matplotlib import pyplot as plt

from examples.setup_simulations import *
from robotics_utilities.iiwa_controller.utils import (
    create_iiwa_controller_plant)


from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.model_paths import models_dir

q_model_path = os.path.join(models_dir, 'q_sys', 'iiwa.yml')


# Simulation parameters.
iiwa_name = "iiwa"
h_quasistatic = 0.2
h_mbp = 1e-4

# Robot joint trajectory.
nq_a = 7
q_iiwa_knots = np.zeros((2, 7))
q_iiwa_knots[0] = [0, 0, 0, -1.70, 0, 1.0, 0]
q_iiwa_knots[1] = q_iiwa_knots[0]
q_iiwa_traj = PiecewisePolynomial.FirstOrderHold([0, 2], q_iiwa_knots.T)

q0_dict_str = {iiwa_name: q_iiwa_knots[0]}
gravity = np.array([0, 0, -10.])

F_WB = np.zeros((2, 3))
F_WB[1] = [0, 0, -100.]
F_WB_traj = PiecewisePolynomial.FirstOrderHold(
    [0, q_iiwa_traj.end_time() / 2], F_WB.T)


def run_comparison(is_visualizing: bool, real_time_rate: float):
    q_parser = QuasistaticParser(q_model_path)

    # Quasistatic.
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_parser=q_parser,
        h=h_quasistatic,
        backend=QuasistaticSystemBackend.PYTHON,
        q_a_traj_dict_str={iiwa_name: q_iiwa_traj},
        q0_dict_str=q0_dict_str,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate,
        body_name="iiwa_link_7",
        F_WB_traj=F_WB_traj)

    # MBP.
    # create controller system for robot.
    plant_robot, _ = create_iiwa_controller_plant(gravity)
    controller_robot = RobotInternalController(
        plant_robot=plant_robot,
        joint_stiffness=q_parser.robot_stiffness_dict[iiwa_name],
        controller_mode="impedance")
    robot_controller_dict = {iiwa_name: controller_robot}

    loggers_dict_mbp_str = run_mbp_sim(
        model_directive_path=q_parser.model_directive_path,
        object_sdf_paths=dict(),
        q_a_traj_dict={iiwa_name: q_iiwa_traj},
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict=q_parser.robot_stiffness_dict,
        robot_controller_dict=robot_controller_dict,
        h=h_mbp,
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
    iiwa_log_mbp = loggers_dict_mbp_str[iiwa_name]
    q_iiwa_mbp = iiwa_log_mbp.data().T[:, :nq]
    t_mbp = iiwa_log_mbp.sample_times()

    iiwa_log_qs = loggers_dict_quasistatic_str[iiwa_name]
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
