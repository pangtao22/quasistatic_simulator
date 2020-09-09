import numpy as np

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
U = np.eye(n_c) * 0.8


tau_ext = np.array([0., -10])
r = 0.1

Kq_a = np.eye(n_a) * 1000


def CalcPhi(q):
    """
    Computes signed distance function for contacts.
    :param q: q: q = [qu, qa]
    :return:
    """
    q_u = q[:n_u]
    q_a = q[n_u:]

    x_c, y_c = q_u
    x_l, x_r, y_g = q_a
    return np.array([x_c - x_l - r, x_r - x_c - r, y_c - r])

