import numpy as np
from pydrake.all import RotationMatrix, GurobiSolver, MosekSolver


def get_rotation_matrix_from_normal(normal):
    n = normal / np.linalg.norm(normal)
    R = np.eye(3)
    R[:, 2] = n
    if np.linalg.norm(n[:2]) < 1e-6:
        R[:, 0] = [0, n[2], -n[1]]
    else:
        R[:, 0] = [n[1], -n[0], 0]
    R[:, 0] /= np.linalg.norm(R[:, 0])
    R[:, 1] = np.cross(n, R[:, 0])

    return R


def calc_tangent_vectors(normal, nd):
    normal = normal.copy()
    normal /= np.linalg.norm(normal)
    if nd == 2:
        # Makes sure that dC is in the yz plane.
        dC = np.zeros((2, 3))
        dC[0] = np.cross(np.array([1, 0, 0]), normal)
        dC[1] = -dC[0]
    else:
        # R = get_rotation_matrix_from_normal(normal)
        R = RotationMatrix.MakeFromOneVector(normal, 2).matrix()
        dC = np.zeros((nd, 3))

        for i in range(nd):
            theta = 2 * np.pi / nd * i
            dC[i] = [np.cos(theta), np.sin(theta), 0]

        dC = (R.dot(dC.T)).T
    return dC


def is_mosek_gurobi_available():
    solver_grb = GurobiSolver()
    solver_msk = MosekSolver()
    return solver_grb.available() and solver_msk.available()
