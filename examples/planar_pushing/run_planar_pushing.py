import os
import numpy as np

from pydrake.all import RigidTransform, DiagramBuilder, PiecewisePolynomial

from examples.setup_simulation_diagram import (
    run_quasistatic_sim, shift_q_traj_to_start_at_minus_h)
from quasistatic_simulation.quasistatic_simulator import (
    RobotInfo, SimulationSettings, create_plant_with_robots_and_objects)

pkg_path = '/Users/pangtao/PycharmProjects/quasistatic_models'
object_sdf_path = os.path.join(pkg_path, "models", "sphere_yz.sdf")
# object_sdf_path = os.path.join(pkg_path, "models", "box_yz_rotation_big.sdf")
robot_sdf_path = os.path.join(pkg_path, "models", "sphere_yz_actuated.sdf")

#%% sim params
h = 0.1
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h
sim_settings = SimulationSettings(is_quasi_dynamic=True,
                                  is_unconstrained=True,
                                  log_barrier_weight=10000,
                                  time_step=h,
                                  contact_detection_tolerance=np.inf)
gravity = np.array([0, 0, -10.])


Kp = np.array([100, 100], dtype=float)
robot_name = "pusher"
robot_info = RobotInfo(
    sdf_path=robot_sdf_path,
    parent_frame_name=None,
    parent_model_name=None,
    base_frame_name=None,
    X_PB=None,
    joint_stiffness=Kp)

robot_info_dict = {robot_name: robot_info}

#%%
nq_a = 2
qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [0, 0.15]
qa_knots[1] = [0.5, 0.15]
qa_traj = PiecewisePolynomial.FirstOrderHold([0, T * h], qa_knots.T)
q_a_traj_dict_str = {robot_name: qa_traj}

# object
object_name = "sphere_yz"
object_sdf_dict = {object_name: object_sdf_path}
qu0 = np.array([0.5, 0.1])

# initial conditions dict.
q0_dict_str = {object_name: qu0,
               robot_name: qa_knots[0]}


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

#%% construct q and v vectors of MBP from log.
    name_to_model_dict = q_sys.q_sim.get_robot_name_to_model_instance_dict()
    logger_qa = loggers_dict_quasistatic_str[robot_name]
    logger_qu = loggers_dict_quasistatic_str[object_name]
    q_log = np.zeros((T, plant.num_positions()))
    v_log = np.zeros((T, plant.num_velocities()))
    tau_a_log = np.zeros((T - 1, plant.num_actuated_dofs()))

    for name, logger in loggers_dict_quasistatic_str.items():
        model = name_to_model_dict[name]
        for i, j in enumerate(q_sys.q_sim.velocity_indices_dict[model]):
            q_log[:, j] = logger.data().T[:, i]

    v_log[1:, :] = (q_log[1:, :] - q_log[:-1, :]) / h

    for l in range(T - 1):
        qa_l = logger_qa.data().T[l]
        qa_l1_cmd = qa_traj.value((l + 1) * h).squeeze()
        tau_a_log[l] = Kp * (qa_l1_cmd - qa_l)

