import matplotlib.pyplot as plt

from examples.setup_simulation_diagram import (
    run_quasistatic_sim, shift_q_traj_to_start_at_minus_h)
from examples.iiwa_block_stacking.iiwa_block_stacking_mbp import run_mbp_sim
from examples.iiwa_block_stacking.simulation_parameters import *
from examples.log_comparison import (calc_error_integral,
                                     calc_pose_error_integral,
                                     get_angle_from_quaternion)


def run_comparison(h_mbp: float, h_quasistatic: float, is_visualizing: bool):
    # %%
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        robot_info_dict=robot_info_dict,
        object_sdf_paths=object_sdf_paths_dict,
        h=h_quasistatic,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=0.0)

    #%%
    loggers_dict_mbp_str = run_mbp_sim(
        q_traj_iiwa=q_iiwa_traj,
        x_traj_schunk=x_schunk_traj,
        robot_info_dict=robot_info_dict,
        object_sdf_paths=object_sdf_paths_dict,
        q0_dict_str=q0_dict_str,
        gravity=gravity,
        time_step=h_mbp,
        is_visualizing=is_visualizing)

    return loggers_dict_mbp_str, loggers_dict_quasistatic_str, q_sys.plant


def compare_all_models(plant,
                       loggers_dict_mbp_str,
                       loggers_dict_quasistatic_str):
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
            q_knots=q_log_quasistatic,
            t=t_quasistatic,
            q_gt_traj=q_mbp_traj)

        error_dict[model_name] = e

    return error_dict


if __name__ == "__main__":
    h_mbp = 1e-3
    h_quasistatic = 0.1

    loggers_dict_mbp_str, loggers_dict_quasistatic_str, plant = \
        run_comparison(h_mbp=h_mbp, h_quasistatic=h_quasistatic,
                       is_visualizing=True)

    error_dict = compare_all_models(
        plant, loggers_dict_mbp_str, loggers_dict_quasistatic_str)
    print(error_dict)

    #%% IIWA joint angles plot.
    q_iiwa_log_mbp = loggers_dict_mbp_str[iiwa_name].data()[:7].T
    t_mbp = loggers_dict_mbp_str[iiwa_name].sample_times()
    q_iiwa_mbp_traj = PiecewisePolynomial.FirstOrderHold(
        t_mbp, q_iiwa_log_mbp.T)

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

    #%% IIWA, mbp / quasistatic vs. quasistatic.
    e, e_vec, t_e = calc_error_integral(
        q_knots=q_iiwa_log_qs,
        t=t_qs,
        q_gt_traj=q_iiwa_mbp_traj)

    shift_q_traj_to_start_at_minus_h(q_iiwa_traj, 0)
    e2, e_vec2, t_e2 = calc_error_integral(
        q_knots=q_iiwa_log_qs,
        t=t_qs,
        q_gt_traj=q_iiwa_traj)

    plt.xlabel("t [s]")
    plt.plot(t_e, e_vec, label="mbp")
    plt.plot(t_e2, e_vec2, label="quasistatic")
    plt.title("IIWA joint angle error, mbp/quasistatic vs commanded.")
    plt.legend()
    plt.show()

    #%% box0, pose error.
    box_name = "box0"
    q_box_log_quasistatic = loggers_dict_quasistatic_str[box_name].data().T
    q_box_log_mbp = loggers_dict_mbp_str[box_name].data()[:7].T

    (e_angle_box, e_vec_angle_box, t_angle_box, e_xyz_box, e_vec_xyz_box,
     t_xyz_box) = calc_pose_error_integral(
        q_box_log_quasistatic, t_qs, q_box_log_mbp, t_mbp)

    print("box angle integral error", e_angle_box)
    print("box position integral error", e_xyz_box)

    plt.plot(t_angle_box, e_vec_angle_box, label="angle [rad]")
    plt.plot(t_xyz_box, e_vec_xyz_box, label="position [m]")
    plt.title("box0 pose error, mbp vs. quasistatic.")
    plt.xlabel("t [s]")
    plt.legend()
    plt.show()

    #%% box angle, quasistatic vs mbp.
    box_angle_quasistatic = [
        get_angle_from_quaternion(q_i) for q_i in q_box_log_quasistatic[:, :4]]
    box_angle_mbp = [
        get_angle_from_quaternion(q_i) for q_i in q_box_log_mbp[:, :4]]
    plt.plot(t_mbp, box_angle_mbp, label="mbp")
    plt.plot(t_qs, box_angle_quasistatic, label="quasistatic")
    plt.legend()
    plt.title("box0 angle [rad]")
    plt.xlabel("t [s]")
    plt.show()

