import numpy as np

from pydrake.all import RigidTransform, DiagramBuilder, PiecewisePolynomial

from examples.setup_simulation_diagram import (
    run_quasistatic_sim, shift_q_traj_to_start_at_minus_h)
from contact_mode_cluster.first_file import robot_sdf_paths, object_sdf_path
from quasistatic_simulation.quasistatic_simulator import (
    RobotInfo, SimulationSettings, create_plant_with_robots_and_objects)

#%% sim params
h = 0.1
T = 10  # num of time steps to simulate forward.
duration = T * h
sim_settings = SimulationSettings(is_quasi_dynamic=True,
                                  is_unconstrained=True,
                                  log_barrier_weight=5000,
                                  time_step=h)
gravity = np.array([0, 0, -10.])

#%%
# robots.
Kp = np.array([50, 25], dtype=float)
robot_l_name = "robot_left"
X_WL = RigidTransform()
X_WL.set_translation([0, -0.1, 1])
robot_l_info = RobotInfo(
    sdf_path=robot_sdf_paths[0],
    parent_model_name="WorldModelInstance",
    parent_frame_name="WorldBody",
    base_frame_name="link_0",
    X_PB=X_WL,
    joint_stiffness=Kp)

robot_r_name = "robot_right"
X_WR = RigidTransform()
X_WR.set_translation([0, 0.1, 1])
robot_r_info = RobotInfo(
    sdf_path=robot_sdf_paths[1],
    parent_model_name="WorldModelInstance",
    parent_frame_name="WorldBody",
    base_frame_name="link_0",
    X_PB=X_WR,
    joint_stiffness=Kp)

robot_info_dict = {robot_l_name: robot_l_info, robot_r_name: robot_r_info}

nq_a = 2
qa_l_knots = np.zeros((2, nq_a))
qa_l_knots[0] = [-np.pi / 4, -np.pi / 4]
q_robot_l_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_l_knots.T)

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 4]
q_robot_r_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_r_knots.T)

q_a_traj_dict_str = {robot_l_name: q_robot_l_traj,
                     robot_r_name: q_robot_r_traj}
# object
object_name = "sphere"
object_sdf_dict = {object_name: object_sdf_path}
q_u0 = np.array([0, 1.7, 0])

# initial conditions dict.
q0_dict_str = {object_name: q_u0,
               robot_l_name: qa_l_knots[0],
               robot_r_name: qa_r_knots[0]}


#%% run sim.
if __name__ == "__main__":
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        robot_info_dict=robot_info_dict,
        object_sdf_paths=object_sdf_dict,
        h=h,
        gravity=gravity,
        is_visualizing=True,
        real_time_rate=1.0,
        sim_settings=sim_settings)



