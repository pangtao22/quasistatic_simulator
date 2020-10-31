import os

import numpy as np
from pydrake.trajectories import PiecewisePolynomial

Kq_a = np.array([1000, 1000, 1000], dtype=float)
q_a0 = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2])
q_a_final = q_a0 + 0.3
t_final = 5

q_a_traj = PiecewisePolynomial.FirstOrderHold(
    [0, t_final], np.vstack([q_a0, q_a_final]).T)
