import os

import numpy as np
from pydrake.trajectories import PiecewisePolynomial

box3d_big_sdf_path = os.path.join("models", "box_1m.sdf")
box3d_medium_sdf_path = os.path.join("models", "box_0.6m.sdf")
box3d_small_sdf_path = os.path.join("models", "box_0.5m.sdf")

Kq_a = np.array([1000, 1000, 1000], dtype=float)
q_a0 = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2])
q_a_final = q_a0 + 0.5
t_final = 5

q_a_traj = PiecewisePolynomial.FirstOrderHold(
    [0, t_final], np.vstack([q_a0, q_a_final]).T)
