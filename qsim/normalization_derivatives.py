import numpy as np


def calc_normalization_derivatives(q: np.ndarray):
    """
    q is an un-normalized vector of length n. q_bar = q / |q|.
    This function returns Dq_barDq as an (n, n) array.
    """
    n = len(q)
    D = np.zeros((n, n))
    q_norm = np.linalg.norm(q)

    for i in range(n):
        D[i] = -q[i] * q / q_norm**3
        D[i, i] += 1 / q_norm

    return D
