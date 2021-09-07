import numpy as np
from pydrake.trajectories import PiecewisePolynomial

#%%
n_u = 1
n_a = 2
n_c = 1
n_d = np.array([2])
n_f = n_d.sum()

Jn = np.array([[0, 0, 1.]])  # 1 x 3.
Jf = np.array([[1, -1, 0.], [-1, 1, 0]])  # 2 x 3.

Jn_u = Jn[:, :n_u]
Jn_a = Jn[:, n_u:]
Jf_u = Jf[:, :n_u]
Jf_a = Jf[:, n_u:]
E = np.ones((n_f, n_c))

mu = 1.0
U = np.eye(n_c) * mu

M_u = np.eye(n_u) * 1.0
tau_ext = np.zeros(n_u)
Kq_a = np.eye(n_a) * 100


def calc_phi(q):
    """
    Computes signed distance function for contacts.
    :param q: q: q = [qu, qa]
    :return:
    """
    y_a = q[2]
    phi = np.array([y_a])
    assert len(phi) == n_c
    return phi


h = 0.1
dq_max = 1 * h  # m/s * s
impulse_max = 50 * h  # N * s
phi_l = np.array([0.])
P_ext = tau_ext * h




