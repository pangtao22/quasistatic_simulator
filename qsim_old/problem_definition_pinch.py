import numpy as np
from pydrake.trajectories import PiecewisePolynomial

#%%
# q = [qu, qa]
n_u = 2  # num of un-actuated DOFs.
n_a = 3  # num of actuated DOFs.
n_c = 3  # num of contacts.
n_d = np.array([2, 2, 2])  # num of rays per friction cone.
n_f = n_d.sum()
assert n_c == n_d.size

Jn_u = np.array([[1, 0], [-1, 0], [0, 1]], dtype=np.float)
Jn_a = np.array([[-1, 0, 0], [0, 1, 0.], [0, 0, 0]])

Jf_u = np.zeros((n_f, n_u))
Jf_u[:, 0] = [0, 0, 0, 0, 1, -1]
Jf_u[:, 1] = [1, -1, 1, -1, 0, 0]

Jf_a = np.zeros((n_f, n_a))
Jf_a[:, 2] = [-1, 1, -1, 1, 0, 0]

U = np.eye(n_c) * 0.5

M_u = np.eye(n_u) * 1.0
tau_ext = np.array([0., -10])

Kq_a = np.eye(n_a) * 1000

r = 0.1


def calc_phi(q):
    """
    Computes signed distance function for contacts.
    :param q: q: q = [qu, qa]
    :return:
    """
    q_u = q[:n_u]
    q_a = q[n_u:]

    x_c, y_c = q_u
    x_l, x_r, y_g = q_a
    phi = np.array([x_c - x_l - r, x_r - x_c - r, y_c - r])
    assert len(phi) == n_c
    return phi


h = 0.01  # simulation time step
dq_max = 1 * h  # 1m/s
impulse_max = 50 * h  # 50N
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

problem_definition['r'] = r
