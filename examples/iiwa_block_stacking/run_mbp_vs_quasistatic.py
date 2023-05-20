import matplotlib.pyplot as plt
from examples.iiwa_block_stacking.simulation_parameters import *
from examples.log_comparison import (
    calc_error_integral,
    calc_pose_error_integral,
    get_angle_from_quaternion,
)
from examples.setup_simulations import (
    run_quasistatic_sim,
    run_mbp_sim,
    shift_q_traj_to_start_at_minus_h,
)
from pydrake.all import PidController, StartMeshcat
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from robotics_utilities.iiwa_controller.robot_internal_controller import (
    RobotInternalController,
)


def run_comparison(h_mbp: float, h_quasistatic: float, is_visualizing: bool):
    q_parser = QuasistaticParser(q_model_path)
    q_parser.set_sim_params(
        h=h_quasistatic,
        use_free_solvers=True,
    )
    meshcat = None
    if is_visualizing:
        meshcat = StartMeshcat()

    # Quasistatic
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_parser=q_parser,
        backend=QuasistaticSystemBackend.CPP,
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        is_visualizing=is_visualizing,
        real_time_rate=0.0,
        meshcat=meshcat,
    )

    q_sys.q_sim.print_solver_info_for_default_params()

    gravity = q_parser.get_gravity()

    # MBP
    plant_robot, _ = create_iiwa_controller_plant(gravity, add_schunk_inertia=True)
    controller_iiwa = RobotInternalController(
        plant_robot=plant_robot,
        joint_stiffness=q_parser.robot_stiffness_dict[iiwa_name],
        controller_mode="impedance",
    )
    # damping calculated for critical of a 2nd order system with the finger's
    # mass.
    controller_schunk = PidController(
        q_parser.robot_stiffness_dict[schunk_name], np.zeros(2), np.ones(2) * 20
    )
    robot_controller_dict = {
        iiwa_name: controller_iiwa,
        schunk_name: controller_schunk,
    }

    loggers_dict_mbp_str = run_mbp_sim(
        model_directive_path=q_parser.model_directive_path,
        object_sdf_paths=q_parser.object_sdf_paths,
        q_a_traj_dict=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict=q_parser.robot_stiffness_dict,
        robot_controller_dict=robot_controller_dict,
        h=h_mbp,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=0,
        meshcat=meshcat,
    )

    return loggers_dict_mbp_str, loggers_dict_quasistatic_str, q_sys.plant


def compare_all_models(plant, loggers_dict_mbp_str, loggers_dict_quasistatic_str):
    error_dict = dict()
    for model_name in q0_dict_str.keys():
        model = plant.GetModelInstanceByName(model_name)
        n_q = plant.num_positions(model)

        q_log_mbp = loggers_dict_mbp_str[model_name].data()[:n_q].T
        t_mbp = loggers_dict_mbp_str[model_name].sample_times()
        q_mbp_traj = PiecewisePolynomial.FirstOrderHold(t_mbp, q_log_mbp.T)

        q_log_quasistatic = loggers_dict_quasistatic_str[model_name].data().T
        t_quasistatic = loggers_dict_quasistatic_str[model_name].sample_times()

        e, _, _ = calc_error_integral(
            q_knots=q_log_quasistatic, t=t_quasistatic, q_gt_traj=q_mbp_traj
        )

        error_dict[model_name] = e

    return error_dict


if __name__ == "__main__":
    h_mbp = 1e-3
    h_quasistatic = 0.1

    loggers_dict_mbp_str, loggers_dict_quasistatic_str, plant = run_comparison(
        h_mbp=h_mbp, h_quasistatic=h_quasistatic, is_visualizing=True
    )

    error_dict = compare_all_models(
        plant, loggers_dict_mbp_str, loggers_dict_quasistatic_str
    )
    print(error_dict)

    # %% IIWA joint angles plot.
    q_iiwa_log_mbp = loggers_dict_mbp_str[iiwa_name].data()[:7].T
    t_mbp = loggers_dict_mbp_str[iiwa_name].sample_times()
    q_iiwa_mbp_traj = PiecewisePolynomial.FirstOrderHold(t_mbp, q_iiwa_log_mbp.T)

    q_iiwa_log_qs = loggers_dict_quasistatic_str[iiwa_name].data()[:7].T
    t_qs = loggers_dict_quasistatic_str[iiwa_name].sample_times()

    figure, axes = plt.subplots(7, 1, figsize=(4, 10), dpi=200)
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_title("IIWA joint angles [rad]")
        ax.plot(t_mbp, q_iiwa_log_mbp[:, i], label="mbp")
        ax.plot(t_qs, q_iiwa_log_qs[:, i], label="quasistatic")
        ax.legend()
    plt.xlabel("t [s]")
    plt.show()

    # %% IIWA, mbp / quasistatic vs. quasistatic.
    e, e_vec, t_e = calc_error_integral(
        q_knots=q_iiwa_log_qs, t=t_qs, q_gt_traj=q_iiwa_mbp_traj
    )

    shift_q_traj_to_start_at_minus_h(q_iiwa_traj, 0)
    e2, e_vec2, t_e2 = calc_error_integral(
        q_knots=q_iiwa_log_qs, t=t_qs, q_gt_traj=q_iiwa_traj
    )

    plt.xlabel("t [s]")
    plt.plot(t_e, e_vec, label="mbp")
    plt.plot(t_e2, e_vec2, label="quasistatic")
    plt.title("IIWA joint angle error, mbp/quasistatic vs commanded.")
    plt.legend()
    plt.show()

    # %% box0, pose error.
    box_name = "box0"
    q_box_log_quasistatic = loggers_dict_quasistatic_str[box_name].data().T
    q_box_log_mbp = loggers_dict_mbp_str[box_name].data()[:7].T

    (
        e_angle_box,
        e_vec_angle_box,
        t_angle_box,
        e_xyz_box,
        e_vec_xyz_box,
        t_xyz_box,
    ) = calc_pose_error_integral(q_box_log_quasistatic, t_qs, q_box_log_mbp, t_mbp)

    print("box angle integral error", e_angle_box)
    print("box position integral error", e_xyz_box)

    plt.plot(t_angle_box, e_vec_angle_box, label="angle [rad]")
    plt.plot(t_xyz_box, e_vec_xyz_box, label="position [m]")
    plt.title("box0 pose error, mbp vs. quasistatic.")
    plt.xlabel("t [s]")
    plt.legend()
    plt.show()

    # %% box angle, quasistatic vs mbp.
    box_angle_quasistatic = [
        get_angle_from_quaternion(q_i) for q_i in q_box_log_quasistatic[:, :4]
    ]
    box_angle_mbp = [get_angle_from_quaternion(q_i) for q_i in q_box_log_mbp[:, :4]]
    plt.plot(t_mbp, box_angle_mbp, label="mbp")
    plt.plot(t_qs, box_angle_quasistatic, label="quasistatic")
    plt.legend()
    plt.title("box0 angle [rad]")
    plt.xlabel("t [s]")
    plt.show()
