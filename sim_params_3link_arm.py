import os

import numpy as np
from pydrake.trajectories import PiecewisePolynomial

box3d_sdf_path = os.path.join("models", "box.sdf")

Kq_a = np.array([1000, 1000, 1000], dtype=float)
q_a0 = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2])
q_u0 = np.array([1, 0, 0, 0, 0, 1.7, 0.5])
q0 = np.hstack([q_u0, q_a0])

t_final = 5
q_a_final = q_a0 + 0.5
q_a_traj = PiecewisePolynomial.FirstOrderHold(
    [0, t_final], np.vstack([q_a0, q_a_final]).T)