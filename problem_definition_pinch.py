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


def CalcE(n_d, n_c):
    E = np.zeros((n_d.sum(), n_c))
    i_start = 0
    for i in range(n_c):
        i_end = i_start + n_d[i]
        E[i_start: i_end, i] = 1
        i_start += n_d[i]
    return E


E = CalcE(n_d, n_c)
U = np.eye(n_c) * 0.5

M_u = np.eye(n_u) * 1.0
tau_ext = np.array([0., -10])
r = 0.1

Kq_a = np.eye(n_a) * 1000


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

# define actuated trajectory
q0 = np.array([0, r, -1.06*r, 1.06*r, 0])

qa_knots = np.zeros((4, 3))
qa_knots[0] = q0[2:]
qa_knots[1] = [-0.9*r, 0.9*r, 0]
qa_knots[2] = [-0.9*r, 0.9*r, -0.03]
qa_knots[3] = [-0.9*r, 0.9*r, -0.03]

n_steps = 35
t_knots = [0, 8*h, (8 + 15)*h, n_steps * h]
q_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)

t_contact_mode_change = [0.03, 0.13, 0.23]


