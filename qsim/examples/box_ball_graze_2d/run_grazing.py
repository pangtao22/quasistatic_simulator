import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from qsim.examples.setup_simulations import run_quasistatic_sim
from qsim_old.problem_definition_graze import problem_definition
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.model_paths import models_dir

# %%
h = problem_definition["h"]

parser = QuasistaticParser(
    os.path.join(models_dir, "q_sys", "ball_grazing_2d.yml")
)
parser.set_quasi_dynamic(True)

# model names
robot_name = "ball"
object_name = "box"
assert np.allclose(
    parser.robot_stiffness_dict[robot_name],
    problem_definition["Kq_a"].diagonal(),
)

# trajectory and initial conditions
nq_a = 2

qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [0, 0.1]
qa_knots[1] = [0.1, -0.1]

n_steps = 10
t_knots = [0, n_steps * h]
qa_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)

q_a_traj_dict_str = {robot_name: qa_traj}
qu0 = np.array([0.0])
q0_dict_str = {object_name: qu0, robot_name: qa_knots[0]}

# %%
loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_parser=parser,
    backend=QuasistaticSystemBackend.PYTHON,
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    is_visualizing=True,
    real_time_rate=1.0,
)
