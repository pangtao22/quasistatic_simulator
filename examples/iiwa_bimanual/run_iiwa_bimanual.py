import os
import copy

import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import run_quasistatic_sim
from qsim.parser import (QuasistaticParser, QuasistaticSystemBackend,
                         GradientMode)
from qsim.simulator import ForwardDynamicsMode
from qsim.model_paths import models_dir

#%%
q_model_path = os.path.join(models_dir, 'q_sys', 'iiwa_bimanual_box.yml')
iiwa_l_name = "iiwa_left"
iiwa_r_name = "iiwa_right"
obj_name = "box"

h = 0.05

q_iiwa_l_knots = np.zeros((2, 7))
q_iiwa_r_knots = np.zeros((2, 7))
# q_iiwa_r_knots[:] = [0, np.pi / 2, 0, 0, 0, 0, 0]
# q_iiwa_r_knots[1, 0] = np.pi / 4
q_iiwa_r_knots[:] = [0.11, 1.57, 0, -0.33, 0, 0, 0]
q_iiwa_l_knots[:] = [-0.09, 1.44, 1.27, 0, 0., 0, 0]

q_iiwa_l_trj = PiecewisePolynomial.FirstOrderHold([0, 1], q_iiwa_l_knots.T)
q_iiwa_r_trj = PiecewisePolynomial.FirstOrderHold([0, 1], q_iiwa_r_knots.T)

q_a_traj_dict_str = {iiwa_l_name: q_iiwa_l_trj,
                     iiwa_r_name: q_iiwa_r_trj}

q_u0 = np.array([1., 0, 0, 0, 0.55, 0, 0.315])
q0_dict_str = {
    obj_name: q_u0,
    iiwa_l_name: q_iiwa_l_knots[0],
    iiwa_r_name: q_iiwa_r_knots[0]}


q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(
    h=h,
    forward_mode=ForwardDynamicsMode.kSocpMp,)

run_quasistatic_sim(
    q_parser=q_parser,
    backend=QuasistaticSystemBackend.CPP,
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    is_visualizing=True,
    real_time_rate=1.0)




