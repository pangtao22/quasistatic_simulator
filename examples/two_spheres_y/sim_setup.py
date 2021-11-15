import os
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (RigidTransform, DiagramBuilder, PiecewisePolynomial,
    TrajectorySource, PidController, Multiplexer, ConnectMeshcatVisualizer,
    LogOutput, Simulator)

from examples.setup_simulation_diagram import (
    run_quasistatic_sim, shift_q_traj_to_start_at_minus_h)
from qsim.simulator import (
    QuasistaticSimParameters, create_plant_with_robots_and_objects)
from examples.model_paths import models_dir


#%% sim setup
object_sdf_path = os.path.join(models_dir, "sphere_y.sdf")
model_directive_path = os.path.join(models_dir, "sphere_y_actuated.yml")

h = 0.05
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h
quasistatic_sim_params = QuasistaticSimParameters(
    gravity=np.array([0, 0, 0.]),
    nd_per_contact=2,
    contact_detection_tolerance=np.inf,
    is_quasi_dynamic=True,
    mode='qp_mp')

# robot
Kp = np.array([500], dtype=float)
robot_name = "sphere_y_actuated"
robot_stiffness_dict = {robot_name: Kp}

# object
object_name = "sphere_y"
object_sdf_dict = {object_name: object_sdf_path}

# trajectory and initial contidionts.
nq_a = 1
# qa_knots = np.zeros((3, nq_a))
# qa_knots[0] = [0]
# qa_knots[1] = [0.8]
# qa_knots[2] = qa_knots[1]
# t_knots = [0, duration * 0.7, duration]

qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [0]
qa_knots[1] = [0.8]
t_knots = [0, duration]

qa_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)
q_a_traj_dict_str = {robot_name: qa_traj}

# initial conditions dict.
qu0 = np.array([0.5])
q0_dict_str = {object_name: qu0, robot_name: qa_knots[0]}