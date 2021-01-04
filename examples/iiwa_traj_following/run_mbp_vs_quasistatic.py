import numpy as np

from pydrake.all import PiecewisePolynomial, Simulator
import matplotlib.pyplot as plt

from quasistatic_simulation.setup_simulation_diagram import *
from setup_environments import create_iiwa_plant
from log_comparison import calc_error_integral
from iiwa_controller.iiwa_controller.utils import create_iiwa_controller_plant

# Simulation parameters.
nq_a = 7
gravity = np.array([0, 0, -10.])
Kp_iiwa = np.array([800., 600, 600, 600, 400, 200, 200])

qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [0, 0, 0, -1.70, 0, 1.0, 0]
qa_knots[1] = qa_knots[0] + 0.5

q_iiwa_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=[0, 10], samples=qa_knots.T,
    sample_dot_at_start=np.zeros(nq_a),
    sample_dot_at_end=np.zeros(nq_a))

h_quasistatic = 0.2
h_mbp = 1e-4


def run_comparison(is_visualizing=False, real_time_rate=0.):
    # Quasistatic sim.
    diagram, loggers_dict_quasistatic, q_sys = setup_quasistatic_sim_diagram(
        q_a_traj_list=[q_iiwa_traj],
        Kp_list=[Kp_iiwa],
        setup_environment=create_iiwa_plant,
        object_sdf_paths=[],
        h=h_quasistatic,
        gravity=gravity,
        is_visualizing=is_visualizing)

    q0_dict = {q_sys.q_sim.models_actuated[0]: qa_knots[0]}
    sim_quasistatic = Simulator(diagram)
    q_sys.set_initial_state(q0_dict)
    sim_quasistatic.Initialize()
    sim_quasistatic.set_target_realtime_rate(real_time_rate)
    sim_quasistatic.AdvanceTo(q_iiwa_traj.end_time())

    # MBP sim.
    (diagram, plant, controller_iiwa, loggers_dict_mbp, robot_model,
        object_models) = setup_mbp_sim_diagram(
        q_a_traj=q_iiwa_traj,
        Kp_a=Kp_iiwa,
        object_sdf_paths=[],
        setup_environment=create_iiwa_plant,
        create_controller_plant=create_iiwa_controller_plant,
        h=h_mbp,
        gravity=gravity,
        is_visualizing=is_visualizing)

    sim_mbp = initialize_mbp_diagram(diagram, plant, controller_iiwa, q0_dict)
    sim_mbp.Initialize()
    sim_mbp.set_target_realtime_rate(real_time_rate)
    sim_mbp.AdvanceTo(q_iiwa_traj.end_time())

    # Extracting iiwa configuration logs.
    q_iiwa_log_mbp = loggers_dict_mbp[robot_model].data()[:nq_a].T
    t_mbp = loggers_dict_mbp[robot_model].sample_times()

    q_iiwa_log_quasistatic = loggers_dict_quasistatic[robot_model].data().T
    t_quasistatic = loggers_dict_quasistatic[robot_model].sample_times()

    return q_iiwa_log_mbp, t_mbp, q_iiwa_log_quasistatic, t_quasistatic


if __name__ == "__main__":
    q_iiwa_log_mbp, t_mbp, q_iiwa_log_quasistatic, t_quasistatic = \
        run_comparison(is_visualizing=True, real_time_rate=0.0)

    # Making plots.
    figure, axes = plt.subplots(7, 1, figsize=(4, 10), dpi=200)

    for i, ax in enumerate(axes):
        ax.plot(t_mbp, q_iiwa_log_mbp[:, i])
        ax.plot(t_quasistatic, q_iiwa_log_quasistatic[:, i])

    plt.show()

    # Set q_iiwa_traj to start at t=0.
    shift_q_traj_to_start_at_minus_h(q_iiwa_traj, 0)
    q_mbp_traj = PiecewisePolynomial.FirstOrderHold(t_mbp, q_iiwa_log_mbp.T)

    e1, e_vec1, t_e1 = calc_error_integral(
        q_knots=q_iiwa_log_quasistatic,
        t=t_quasistatic,
        q_gt_traj=q_mbp_traj)
    print("Quasistatic vs MBP", e1)

    e2, e_vec2, t_e2 = calc_error_integral(
        q_knots=q_iiwa_log_mbp,
        t=t_mbp,
        q_gt_traj=q_iiwa_traj)
    print("MBP vs commanded", e2)

    e3, e_vec3, t_e3 = calc_error_integral(
        q_knots=q_iiwa_log_quasistatic,
        t=t_quasistatic,
        q_gt_traj=q_iiwa_traj)
    print("Quasistatic vs commanded", e3)

    e4, e_vec4, t_e4 = calc_error_integral(
        q_knots=q_iiwa_log_mbp,
        t=t_mbp,
        q_gt_traj=q_mbp_traj)
    print("MBP vs itself", e4)

    #%%
    plt.plot(t_e1, e_vec1, label="quasistatic")
    plt.plot(t_e2, e_vec2, label="mbp")
    plt.legend()
    plt.show()
