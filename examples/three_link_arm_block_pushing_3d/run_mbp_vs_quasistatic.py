import matplotlib.pyplot as plt

from examples.setup_simulation_diagram import *
from examples.log_comparison import *
from examples.setup_environments import (
    box3d_big_sdf_path,
    create_3link_arm_plant_with_multiple_objects,
    create_3link_arm_controller_plant)

# Simulation parameters.
robot_name = "three_link_arm"
box_name = "box0"

# Simulation parameters.
nq_a = 3
gravity = np.array([0, 0, -10.])
Kp_robot = np.array([1000, 1000, 1000], dtype=float)

qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [np.pi / 2, -np.pi / 2, -np.pi / 2]
qa_knots[1] = qa_knots[0] + 0.3

q_robot_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=[0, 10], samples=qa_knots.T,
    sample_dot_at_start=np.zeros(nq_a),
    sample_dot_at_end=np.zeros(nq_a))

q_u0 = np.array([1, 0, 0, 0, 0.1, 1.7, 0.5])
q0_dict_str = {robot_name: qa_knots[0], box_name: q_u0}

h_quasistatic = 0.01
h_mbp = 1e-4


def run_comparison(is_visualizing=False, real_time_rate=0.0):
    #%% Quasistatic
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_a_traj_dict_str={robot_name: q_robot_traj},
        q0_dict_str=q0_dict_str,
        Kp_list=[Kp_robot],
        setup_environment=create_3link_arm_plant_with_multiple_objects,
        object_sdf_paths=[box3d_big_sdf_path],
        h=h_quasistatic,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate)

    #%% MBP
    loggers_dict_mbp_str = run_mbp_sim(
        q_a_traj=q_robot_traj,
        q0_dict_str=q0_dict_str,
        Kp_a=Kp_robot,
        object_sdf_paths=[box3d_big_sdf_path],
        setup_environment=create_3link_arm_plant_with_multiple_objects,
        create_controller_plant=create_3link_arm_controller_plant,
        h=h_mbp,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate)

    # Extracting iiwa configuration logs.
    q_box_log_mbp = loggers_dict_mbp_str[box_name].data()[:7].T
    q_robot_log_mbp = loggers_dict_mbp_str[robot_name].data()[:nq_a].T
    t_mbp = loggers_dict_mbp_str[robot_name].sample_times()

    q_robot_log_quasistatic = loggers_dict_quasistatic_str[robot_name].data().T
    q_box_log_quasistatic = loggers_dict_quasistatic_str[box_name].data().T
    t_quasistatic = loggers_dict_quasistatic_str[robot_name].sample_times()

    return (q_robot_log_mbp, q_box_log_mbp, t_mbp,
            q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic,
            q_sys)


def calc_integral_errors(q_robot_log_mbp, q_box_log_mbp, t_mbp,
     q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic):
    # Set q_iiwa_traj to start at t=0.
    shift_q_traj_to_start_at_minus_h(q_robot_traj, 0)

    # Convert mbp knot points to a polynomial.
    qa_mbp_traj = PiecewisePolynomial.ZeroOrderHold(t_mbp, q_robot_log_mbp.T)

    # Robot.
    e_robot, e_vec_robot, t_e_robot = calc_error_integral(
        q_knots=q_robot_log_quasistatic,
        t=t_quasistatic,
        q_gt_traj=qa_mbp_traj)

    # Object orientation.
    quaternion_box_mbp_traj = convert_quaternion_array_to_eigen_quaternion_traj(
        q_box_log_mbp[:, :4], t_mbp)

    e_angle_box, e_vec_angle_box, t_angle_box = calc_quaternion_error_integral(
        q_list=q_box_log_quasistatic[:, :4],
        t=t_quasistatic,
        q_traj=quaternion_box_mbp_traj)

    # Object position.
    xyz_box_mbp_traj = PiecewisePolynomial.FirstOrderHold(
        t_mbp, q_box_log_mbp[:, 4:].T)
    e_xyz_box, e_vec_xyz_box, t_xyz_box = calc_error_integral(
        q_knots=q_box_log_quasistatic[:, 4:],
        t=t_quasistatic,
        q_gt_traj=xyz_box_mbp_traj)

    return (e_robot, e_vec_robot, t_e_robot,
            e_angle_box, e_vec_angle_box, t_angle_box,
            e_xyz_box, e_vec_xyz_box, t_xyz_box)


if __name__ == "__main__":
    (q_robot_log_mbp, q_box_log_mbp, t_mbp,
     q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic, q_sys) = \
        run_comparison(is_visualizing=True, real_time_rate=0.0)

    figure, axes = plt.subplots(nq_a, 1, figsize=(4, 10), dpi=200)
    for i, ax in enumerate(axes):
        ax.plot(t_mbp, q_robot_log_mbp[:, i], label="mbp")
        ax.plot(t_quasistatic, q_robot_log_quasistatic[:, i],
                label="quasistatic")
        ax.legend()
    plt.show()

    (e_robot, e_vec_robot, t_e_robot,
     e_angle_box, e_vec_angle_box, t_angle_box,
     e_xyz_box, e_vec_xyz_box, t_xyz_box) = calc_integral_errors(
        q_robot_log_mbp, q_box_log_mbp, t_mbp,
        q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic)

    #%% Robot.
    print('Quasistatic vs mbp, robot', e_robot)

    #%% Object orientation.
    print("Quasistatic vs mbp, object angle", e_angle_box)
    plt.plot(t_angle_box, e_vec_angle_box)
    plt.title("Angle difference [rad]")
    plt.show()

    #%% Object position.
    print("Quasistatic vs mbp, object position", e_xyz_box)
    plt.plot(t_xyz_box, e_vec_xyz_box)
    plt.title("Position difference [m]")
    plt.show()


