import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import (
    run_quasistatic_sim)
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.model_paths import models_dir


#%% sim setup
q_model_path = os.path.join(models_dir, 'q_sys', 'allegro_hand_and_sphere.yml')

h = 0.1
duration = 2

hand_name = 'allegro_hand_right'
object_name = 'sphere'
nq_a = 16

qa_knots = np.zeros((2, nq_a))
qa_knots[1] += 1.0
qa_knots[1, 0] = 0
qa_knots[1, 8] = 0
qa_knots[1, 12] = 0
qa_traj = PiecewisePolynomial.FirstOrderHold(
    [0, duration], qa_knots.T)

qu0 = np.zeros(7)
qu0[:4] = [1, 0, 0, 0]
qu0[4:] = [-0.12, 0.01, 0.07]

q_a_traj_dict_str = {hand_name: qa_traj}
q0_dict_str = {hand_name: qa_knots[0],
               object_name: qu0}

q_parser = QuasistaticParser(q_model_path)
q_parser.set_quasi_dynamic(True)

loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_parser=q_parser,
    h=h,
    backend=QuasistaticSystemBackend.CPP,
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    is_visualizing=True,
    real_time_rate=1.0)



