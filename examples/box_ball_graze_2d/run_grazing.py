import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulation_diagram import run_quasistatic_sim
from qsim.simulator import QuasistaticSimParameters
from qsim_old.problem_definition_graze import problem_definition
from qsim.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "box_y.sdf")
model_directive_path = os.path.join(models_dir, "box_ball_graze_2d.yml")

#%%
h = problem_definition['h']
quasistatic_sim_params = QuasistaticSimParameters(
    gravity=np.array([0, 0, 0.]),
    nd_per_contact=2,
    contact_detection_tolerance=np.inf,
    is_quasi_dynamic=True,
    requires_grad=True)

# robot
Kp = problem_definition['Kq_a'].diagonal()
robot_name = 'ball'
robot_stiffness_dict = {robot_name: Kp}

# object
object_name = "box"
object_sdf_dict = {object_name: object_sdf_path}

# trajectory and initial conditions
nq_a = 2

qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [0, 0.1]
qa_knots[1] = [0.1, -0.1]

n_steps = 10
t_knots = [0, n_steps * h]
qa_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)

q_a_traj_dict_str = {robot_name: qa_traj}
qu0 = np.array([0.])
q0_dict_str = {object_name: qu0,
               robot_name: qa_knots[0]}

#%%
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
