import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import run_quasistatic_sim
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.simulator import ForwardDynamicsMode, GradientMode
from qsim.model_paths import models_dir


#%% sim setup
q_model_path = os.path.join(models_dir, 'q_sys', 'two_spheres_xyz.yml')

h = 0.1
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h

# robot
robot_name = "sphere_xyz_actuated"
object_name = "sphere_xyz"
r_robot = 0.1
r_obj = 0.5


# trajectory and initial conditions.
nq_a = 3
theta = np.pi / 6

qa_knots = np.zeros((3, nq_a))
qa_knots[0] = [-np.cos(theta) * (r_robot + r_obj),
               -np.sin(theta) * (r_robot + r_obj),
               r_obj]
qa_knots[1] = [np.cos(theta) * 1.0,
               np.sin(theta) * 1.0,
               r_obj]
qa_knots[2] = qa_knots[1]
qa_traj = PiecewisePolynomial.FirstOrderHold([0, duration * 0.8, duration],
                                             qa_knots.T)
q_a_traj_dict_str = {robot_name: qa_traj}
qu0 = np.array([0., 0., r_obj])
q0_dict_str = {object_name: qu0,
               robot_name: qa_knots[0]}


q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(
    h=h,
    is_quasi_dynamic=True,
    forward_mode=ForwardDynamicsMode.kQpMp,
    log_barrier_weight=100)


loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_parser=q_parser,
    backend=QuasistaticSystemBackend.CPP,
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    is_visualizing=True,
    real_time_rate=1.0)




