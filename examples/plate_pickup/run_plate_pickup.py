import os
import numpy as np

from pydrake.all import RigidTransform, DiagramBuilder, PiecewisePolynomial

from examples.setup_simulation_diagram import (
    run_quasistatic_sim, shift_q_traj_to_start_at_minus_h)
from qsim.simulator import QuasistaticSimParameters
from examples.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "plate.sdf")
model_directive_path = os.path.join(models_dir, "plate.yml")

#%% sim setup
h = 0.2
T = int(round(12 / h))  # num of time steps to simulate forward.
duration = T * h
sim_settings = QuasistaticSimParameters(
    gravity=np.array([0, 0, -10.]),
    nd_per_contact=2,
    contact_detection_tolerance=1.0,
    is_quasi_dynamic=True)

# robots.
Kp = np.array([50, 50, 100, 50, 50], dtype=float)
robot_stiffness_dict = {"gripper": Kp}

# object
object_name = "plate"
object_sdf_dict = {object_name: object_sdf_path}

# trajectory and initial conditions.
nq_a = 5
qa_knots = np.zeros((7, nq_a))
# x, y, z, dy1, dy2
qa_knots[0] = [0.32, 0.2, -0.6, -0.05, 0.05]
qa_knots[1] = [0.22, 0.05, -0.6, -0.04, 0.04]
qa_knots[2] = [0.18, 0.05, -1.0, -0.02, 0.02]
qa_knots[3] = [0.05, 0.03, -1.2, -0.02, 0.02]
qa_knots[4] = [0.05, 0.06, -1.2, -0.01, 0.01]
qa_knots[5] = [0.05, 0.2, -1.2, -0.00, 0.00]
qa_knots[6] = [0.32, 0.5, -1.2, -0.00, 0.00]
q_robot_traj = PiecewisePolynomial.FirstOrderHold(
    [0, 2, 4, 6, 8, 10, 12], qa_knots.T)

q_a_traj_dict_str = {"gripper": q_robot_traj}

q_u0 = np.array([0, 0.1, 0])

q0_dict_str = {"plate": q_u0,
               "gripper": qa_knots[0]}


#%% run sim.
if __name__ == "__main__":
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        model_directive_path=model_directive_path,
        object_sdf_paths=object_sdf_dict,
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict=robot_stiffness_dict,
        h=h,
        sim_params=sim_settings,
        is_visualizing=True,
        real_time_rate=1.0)
