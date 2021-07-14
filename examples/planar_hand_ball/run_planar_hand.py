import numpy as np

from pydrake.all import RigidTransform, DiagramBuilder, PiecewisePolynomial

from examples.setup_simulation_diagram import (
    run_quasistatic_sim, shift_q_traj_to_start_at_minus_h)
from contact_mode_cluster.first_file import robot_sdf_paths, object_sdf_path
from quasistatic_simulation.quasistatic_simulator import (
    RobotInfo, SimulationSettings, create_plant_with_robots_and_objects)

#%% sim params
h = 0.2
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h
sim_settings = SimulationSettings(is_quasi_dynamic=True,
                                  is_unconstrained=True,
                                  log_barrier_weight=100,
                                  time_step=h,
                                  contact_detection_tolerance=1.0)
gravity = np.array([0, 0, -10.])

#%%
# robots.
Kp = np.array([50, 25], dtype=float)
robot_l_name = "robot_left"
X_WL = RigidTransform()
X_WL.set_translation([0, -0.1, 2])
robot_l_info = RobotInfo(
    sdf_path=robot_sdf_paths[0],
    parent_model_name="WorldModelInstance",
    parent_frame_name="WorldBody",
    base_frame_name="link_0",
    X_PB=X_WL,
    joint_stiffness=Kp)

robot_r_name = "robot_right"
X_WR = RigidTransform()
X_WR.set_translation([0, 0.1, 2])
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
q_u0 = np.array([0, 2.35, 0])

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
        sim_settings=sim_settings,
        nd_per_contact=2)

#%% look into the plant.
    plant = q_sys.plant
    for model in q_sys.q_sim.models_all:
        print(model, plant.GetModelInstanceName(model),
              q_sys.q_sim.velocity_indices_dict[model])


#%% index for tau_a.
    indices = []
    for model in q_sys.q_sim.models_actuated:
        indices += q_sys.q_sim.velocity_indices_dict[model].tolist()
    indices.sort()
    indices_map = {j: i for i, j in enumerate(indices)}

#%% construct q and v vectors of MBP from log.
    name_to_model_dict = q_sys.q_sim.get_robot_name_to_model_instance_dict()
    logger_qu = loggers_dict_quasistatic_str[object_name]
    q_log = np.zeros((T, plant.num_positions()))
    v_log = np.zeros((T, plant.num_velocities()))
    tau_a_log = np.zeros((T - 1, plant.num_actuated_dofs()))

    for name, logger in loggers_dict_quasistatic_str.items():
        model = name_to_model_dict[name]
        for i, j in enumerate(q_sys.q_sim.velocity_indices_dict[model]):
            q_log[:, j] = logger.data().T[:, i]

    v_log[1:, :] = (q_log[1:, :] - q_log[:-1, :]) / h

    for name in robot_info_dict.keys():
        model = name_to_model_dict[name]
        logger_qa = loggers_dict_quasistatic_str[name]
        idx_v = q_sys.q_sim.velocity_indices_dict[model]
        idx_tau_a = [indices_map[i] for i in idx_v]
        for l in range(T - 1):
            qa_l = logger_qa.data().T[l]
            qa_l1_cmd = q_a_traj_dict_str[name].value((l + 1) * h).squeeze()
            tau_a_log[l][idx_tau_a] = Kp * (qa_l1_cmd - qa_l)
