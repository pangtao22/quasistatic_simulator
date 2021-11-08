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


problem_definition = dict()
problem_definition['n_u'] = n_u
problem_definition['n_a'] = n_a
problem_definition['n_c'] = n_c
problem_definition['n_d'] = n_d
problem_definition['n_f'] = n_f
problem_definition['Jn_u'] = Jn_u
problem_definition['Jn_a'] = Jn_a
problem_definition['Jf_u'] = Jf_u
problem_definition['Jf_a'] = Jf_a
problem_definition['U'] = U
problem_definition['M_u'] = M_u
problem_definition['tau_ext'] = tau_ext
problem_definition['Kq_a'] = Kq_a
problem_definition['calc_phi'] = calc_phi
problem_definition['h'] = h
problem_definition['dq_max'] = dq_max
problem_definition['impulse_max'] = impulse_max
problem_definition['P_ext'] = P_ext
