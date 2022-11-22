import matplotlib.pyplot as plt
from examples.log_comparison import *
from examples.setup_simulations import shift_q_traj_to_start_at_minus_h
from examples.three_link_arm_block_pushing.utils import *

# box initial state.
q_u0 = np.array([1.7, 0.5, 0])
q0_dict_str = {robot_name: qa_knots[0], box_name: q_u0}


def calc_integral_errors(
    q_robot_log_mbp,
    q_box_log_mbp,
    t_mbp,
    q_robot_log_quasistatic,
    q_box_log_quasistatic,
    t_quasistatic,
):
    shift_q_traj_to_start_at_minus_h(q_robot_traj, 0)
    qa_mbp_traj = PiecewisePolynomial.ZeroOrderHold(t_mbp, q_robot_log_mbp.T)

    # Quasistatic vs MBP, robot.
    e_robot, e_vec_robot, t_e_robot = calc_error_integral(
        q_knots=q_robot_log_quasistatic, t=t_quasistatic, q_gt_traj=qa_mbp_traj
    )

    # Quasistatic vs MBP, object angle.
    angle_mbp = q_box_log_mbp[:, 2].T
    angle_mbp = angle_mbp.reshape((1, angle_mbp.size))
    angle_box_mbp_traj = PiecewisePolynomial.FirstOrderHold(t_mbp, angle_mbp)
    e_angle_box, e_vec_angle_box, t_angle_box = calc_error_integral(
        q_knots=q_box_log_quasistatic[:, 2],
        t=t_quasistatic,
        q_gt_traj=angle_box_mbp_traj,
    )
    # Quasistatic vs MBP, object position.
    xyz_box_mbp_traj = PiecewisePolynomial.FirstOrderHold(
        t_mbp, q_box_log_mbp[:, :2].T
    )
    e_xyz_box, e_vec_xyz_box, t_xyz_box = calc_error_integral(
        q_knots=q_box_log_quasistatic[:, :2],
        t=t_quasistatic,
        q_gt_traj=xyz_box_mbp_traj,
    )

    return (
        e_robot,
        e_vec_robot,
        t_e_robot,
        e_angle_box,
        e_vec_angle_box,
        t_angle_box,
        e_xyz_box,
        e_vec_xyz_box,
        t_xyz_box,
    )


if __name__ == "__main__":
    (
        q_robot_log_mbp,
        q_box_log_mbp,
        t_mbp,
        q_robot_log_quasistatic,
        q_box_log_quasistatic,
        t_quasistatic,
        q_sys,
    ) = run_mbp_quasistatic_comparison(
        q_model_path_2d, q0_dict_str, is_visualizing=True, real_time_rate=0.0
    )
    #%%
    figure, axes = plt.subplots(nq_a, 1, figsize=(4, 10), dpi=200)
    axes[0].set_title("Joint angles")
    for i, ax in enumerate(axes):
        ax.plot(t_mbp, q_robot_log_mbp[:, i], label="mbp")
        ax.plot(
            t_quasistatic, q_robot_log_quasistatic[:, i], label="quasistatic"
        )
        ax.set_ylabel("[rad]")
        ax.legend()
    plt.xlabel("t [s]")
    plt.show()

    (
        e_robot,
        e_vec_robot,
        t_e_robot,
        e_angle_box,
        e_vec_angle_box,
        t_angle_box,
        e_xyz_box,
        e_vec_xyz_box,
        t_xyz_box,
    ) = calc_integral_errors(
        q_robot_log_mbp,
        q_box_log_mbp,
        t_mbp,
        q_robot_log_quasistatic,
        q_box_log_quasistatic,
        t_quasistatic,
    )

    # %%
    print("Quasistatic vs MBP, robot", e_robot)

    # %% Orientation (angle).
    print("Quasistatic vs MBP, object angle", e_angle_box)
    plt.plot(t_angle_box, e_vec_angle_box)
    plt.title("Box angle difference, mbp vs. quasistatic [rad]")
    plt.xlabel("t [s]")
    plt.show()

    # %% Position.
    print("Quasistatic vs MBP, object position", e_xyz_box)
    plt.plot(t_xyz_box, e_vec_xyz_box)
    plt.title("Box position difference, mbp vs. quasistatic [m]")
    plt.xlabel("t [s]")
    plt.show()
