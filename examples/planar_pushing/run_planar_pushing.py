import os
import numpy as np

from pydrake.all import RigidTransform, DiagramBuilder, PiecewisePolynomial

from examples.setup_simulation_diagram import (
    run_quasistatic_sim, shift_q_traj_to_start_at_minus_h)
from quasistatic_simulation.quasistatic_simulator import (
    QuasistaticSimParameters, create_plant_with_robots_and_objects)
from examples.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "sphere_yz.sdf")
model_directive_path = os.path.join(models_dir,
                                    "sphere_yz_actuated_and_ground.yml")

#%% sim params
h = 0.05
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h
quasistatic_sim_params = QuasistaticSimParameters(
    gravity=np.array([0, 0, -10.]),
    nd_per_contact=2,
    contact_detection_tolerance=np.inf,
    is_quasi_dynamic=True,
    is_unconstrained=False)

Kp = np.array([100, 100], dtype=float)
robot_name = "sphere_yz_actuated"
robot_stiffness_dict = {robot_name: Kp}

#%%
nq_a = 2
qa_knots = np.zeros((3, nq_a))
qa_knots[0] = [0, 0.15]
qa_knots[1] = [0.8, 0.15]
qa_knots[2] = qa_knots[1]
qa_traj = PiecewisePolynomial.FirstOrderHold([0, duration * 0.8, duration],
                                             qa_knots.T)
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
        model_directive_path=model_directive_path,
        object_sdf_paths=object_sdf_dict,
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict={robot_name: Kp},
        h=h,
        sim_params=quasistatic_sim_params,
        is_visualizing=True,
        real_time_rate=1.0)


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

