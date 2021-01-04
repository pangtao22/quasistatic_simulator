from examples.setup_simulation_diagram import *
from examples.log_comparison import *
from examples.setup_environments import (
    box2d_big_sdf_path,
    create_3link_arm_plant_with_multiple_objects,
    create_3link_arm_controller_plant)

is_visualizing = True
real_time_rate = 1.0

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

q_u0 = np.array([1.7, 0.5, 0])


#%% Quasistatic
h_quasistatic = 0.01
diagram, loggers_dict_quasistatic, q_sys = run_quasistatic_sim(
    q_a_traj_dict_str=[q_robot_traj],
    Kp_list=[Kp_robot],
    setup_environment=create_3link_arm_plant_with_multiple_objects,
    object_sdf_paths=[box2d_big_sdf_path],
    h=h_quasistatic,
    gravity=gravity,
    is_visualizing=is_visualizing)


q0_dict = {q_sys.q_sim.models_actuated[0]: qa_knots[0],
           q_sys.q_sim.models_unactuated[0]: q_u0}
sim_quasistatic = Simulator(diagram)
q_sys.set_initial_state(q0_dict)
sim_quasistatic.Initialize()
sim_quasistatic.set_target_realtime_rate(real_time_rate)
sim_quasistatic.AdvanceTo(q_robot_traj.end_time())


#%% MBP
h_mbp = 1e-4
(diagram, plant, controller_iiwa, loggers_dict_mbp, robot_model,
    object_models) = run_mbp_sim(
    q_a_traj=q_robot_traj,
    Kp_a=Kp_robot,
    object_sdf_paths=[box2d_big_sdf_path],
    setup_environment=create_3link_arm_plant_with_multiple_objects,
    create_controller_plant=create_3link_arm_controller_plant,
    h=h_mbp,
    gravity=gravity,
    is_visualizing=is_visualizing)

sim_mbp = initialize_mbp_diagram(diagram, plant, controller_iiwa, q0_dict)

# %%
sim_mbp.Initialize()
sim_mbp.set_target_realtime_rate(real_time_rate)
sim_mbp.AdvanceTo(q_robot_traj.end_time())


#%%
import matplotlib.pyplot as plt
figure, axes = plt.subplots(nq_a, 1, figsize=(4, 10), dpi=200)

q_robot_log_mbp = loggers_dict_mbp[robot_model].data()[:nq_a].T
t_mbp = loggers_dict_mbp[robot_model].sample_times()

q_robot_log_quasistatic = loggers_dict_quasistatic[robot_model].data().T
t_quasistatic = loggers_dict_quasistatic[robot_model].sample_times()

for i, ax in enumerate(axes):
    ax.plot(t_mbp, q_robot_log_mbp[:, i], label="mbp")
    ax.plot(t_quasistatic, q_robot_log_quasistatic[:, i], label="quasistatic")
    ax.legend()

plt.show()

#%%
# convert q_gt_knots to a piecewise polynomial.
shift_q_traj_to_start_at_minus_h(q_robot_traj, 0)
qa_mbp_traj = PiecewisePolynomial.ZeroOrderHold(t_mbp, q_robot_log_mbp.T)

e_robot, e_vec_robot, t_e_robot = calc_error_integral(
    q_knots=q_robot_log_quasistatic,
    t=t_quasistatic,
    q_gt_traj=qa_mbp_traj)
print(e_robot)

#%% Object pose.
box_model = object_models[0]
q_box_log_mbp = loggers_dict_mbp[box_model].data()[:7].T
t_mbp = loggers_dict_mbp[box_model].sample_times()

q_box_log_quasistatic = loggers_dict_quasistatic[box_model].data().T
t_quasistatic = loggers_dict_quasistatic[box_model].sample_times()


#%% Orientation (yaw angle).
yaw_angle_mbp = q_box_log_mbp[:, 2].T
yaw_angle_mbp = yaw_angle_mbp.reshape((1, yaw_angle_mbp.size))
yaw_box_mbp_traj = PiecewisePolynomial.FirstOrderHold(t_mbp, yaw_angle_mbp)
e_quat_box, e_vec_quat_box, t_quat_box = calc_error_integral(
    q_knots=q_box_log_quasistatic[:, 2],
    t=t_quasistatic,
    q_gt_traj=yaw_box_mbp_traj)

print(e_quat_box)
plt.plot(t_quat_box, e_vec_quat_box)
plt.title("Angle difference [rad]")
plt.show()

#%% Position.
xyz_box_mbp_traj = PiecewisePolynomial.FirstOrderHold(t_mbp,
                                                      q_box_log_mbp[:, :2].T)
e_xyz_box, e_vec_xyz_box, t_xyz_box = calc_error_integral(
    q_knots=q_box_log_quasistatic[:, :2],
    t=t_quasistatic,
    q_gt_traj=xyz_box_mbp_traj)

print(e_xyz_box)
plt.plot(t_xyz_box, e_vec_xyz_box)
plt.title("Position difference [m]")
plt.show()

#%%

