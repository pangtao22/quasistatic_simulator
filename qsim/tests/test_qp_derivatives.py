import unittest

import numpy as np
from pydrake.all import OsqpSolver, GurobiSolver

from robotics_utilities.qp_derivatives.qp_derivatives import(
    QpDerivativesKktActive, build_qp_and_solve)

from qsim_cpp import QpDerivativesActive


def get_DzDG_active_from_DzDG(DzDG_vec, active_row_indices):
    n_z, n_G_entries = DzDG_vec.shape
    n_lambda = n_G_entries // n_z
    assert n_G_entries == n_lambda * n_z

    active_indices = []
    for j in range(n_z):
        for i in active_row_indices:
            active_indices.append(j * n_lambda + i)

    return DzDG_vec[:, active_indices]


class TestQpDerivatives(unittest.TestCase):
    def setUp(self):
        """
        The first problem in this test case is
        min. 0.5 * ((x0 - 1)^2 + (x1 + 1)^2) - 1
        s.t. x0 >= 0, x1 >= 0

        Solution is x_star = [1, 0]
        """
        # TODO: this setup function is copied from robotics_utilites. Is there
        # a way to avoid the copying?
        n_z = 2
        self.Q_list = [np.eye(n_z)]
        self.b_list = [-np.array([1., -1])]
        self.G_list = [-np.eye(n_z)]
        self.e_list = [np.zeros(n_z)]

        n_z = 2
        n_lambda = 3
        np.random.seed(2024)
        L = np.random.rand(n_z, n_z) - 1
        self.Q_list.append(L.T.dot(L))
        self.b_list.append(np.random.rand(n_z) - 1)
        self.G_list.append(np.random.rand(n_lambda, n_z))
        self.e_list.append(np.array([0, 0, 10.]))

        gurobi_solver = GurobiSolver()
        if gurobi_solver.available():
            self.solver = GurobiSolver()
        else:
            self.solver = OsqpSolver()

        self.dqp_py = QpDerivativesKktActive()
        self.dqp_cpp = QpDerivativesActive(1e-3)

    def test_derivatives(self):
        lambda_threshold = 1e-3
        atol = 1e-8
        for Q, b, G, e in zip(self.Q_list,
                              self.b_list, self.G_list, self.e_list):
            z_star, lambda_star = build_qp_and_solve(Q, b, G, e, self.solver)
            self.dqp_py.update_problem(
                Q=Q, b=b, G=G, e=e, z_star=z_star, lambda_star=lambda_star,
                lambda_threshold=lambda_threshold)
            self.dqp_cpp.UpdateProblem(
                Q, b, G, e, z_star, lambda_star, lambda_threshold, True)

            DzDe_py = self.dqp_py.calc_DzDe()
            DzDe_cpp = self.dqp_cpp.get_DzDe()

            DzDb_py = self.dqp_py.calc_DzDb()
            DzDb_cpp = self.dqp_cpp.get_DzDb()

            DzdG_vec_py = self.dqp_py.calc_DzDG_vec()
            (DzdG_vec_active_cpp, active_row_indices
             ) = self.dqp_cpp.get_DzDvecG_active()
            DzdG_vec_active_py = get_DzDG_active_from_DzDG(
                DzdG_vec_py, active_row_indices)

            np.testing.assert_allclose(DzDe_py, DzDe_cpp, atol=atol)
            np.testing.assert_allclose(DzDb_py, DzDb_cpp, atol=atol)
            np.testing.assert_allclose(
                DzdG_vec_active_cpp, DzdG_vec_active_py, atol=atol)
