import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from qsim.model_paths import models_dir


#%% sim setup
q_model_path = os.path.join(models_dir, "q_sys", "two_spheres_y.yml")

h = 0.05
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h

# names
robot_name = "sphere_y_actuated"
object_name = "sphere_y"

# trajectory and initial conditions.
nq_a = 1

qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [0]
qa_knots[1] = [0.8]
t_knots = [0, duration]

qa_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)
q_a_traj_dict_str = {robot_name: qa_traj}

# initial conditions dict.
qu0 = np.array([0.5])
q0_dict_str = {object_name: qu0, robot_name: qa_knots[0]}
