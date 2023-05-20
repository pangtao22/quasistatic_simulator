import unittest

import numpy as np
import cvxpy as cp

from qsim_cpp import SocpDerivatives

from pydrake.all import MathematicalProgram, MosekSolver, ScsSolver


class TestSocpDerivativesCpp(unittest.TestCase):
    def setUp(self):
        # Set up a cone program with linear cost b @ z and three SOC
        # constraints, the first two are active and the third is not.
        n_z = 2
        m = 2
        n_c = 3

        # Solve the problem with CVXPY.
        J_cvx = [cp.Parameter((m, n_z)) for _ in range(n_c)]
        J_cvx[0].value = np.eye(n_z)
        J_cvx[1].value = np.eye(n_z) * 1.1
        J_cvx[2].value = np.eye(n_z)

        b_cvx = cp.Parameter(n_z)
        b_cvx.value = np.array([1, 0])

        e_cvx = [cp.Parameter(m) for _ in range(n_c)]
        e_cvx[0].value = np.zeros(m)
        e_cvx[1].value = np.array([0, 2])
        e_cvx[2].value = np.array([0, 1])

        z = cp.Variable(n_z)
        s = [J_cvx[i] @ z + e_cvx[i] for i in range(n_c)]

        constraints = [cp.constraints.second_order.SOC(s_i[0], s_i[1:]) for s_i in s]

        prob = cp.Problem(cp.Minimize(b_cvx @ z), constraints)

        prob.solve(requires_grad=True)

        # extract gradients
        DzDb = np.zeros((n_z, n_z))
        DzDe = np.zeros((n_c, n_z, m))
        DzDvecJ = np.zeros((n_c, n_z, J_cvx[0].size))

        for i in range(n_z):
            dv = np.zeros(n_z)
            dv[i] = 1
            z.gradient = dv
            prob.backward()

            DzDb[i] = b_cvx.gradient

            for i_c in range(n_c):
                DzDe[i_c, i] = e_cvx[i_c].gradient
                DzDvecJ[i_c, i] = np.concatenate(J_cvx[i_c].gradient.transpose())

        self.DzDb = DzDb
        self.DzDe = DzDe
        self.DzDvecJ = DzDvecJ

        self.b_cvx = b_cvx
        self.e_cvx = e_cvx
        self.J_cvx = J_cvx
        self.n_z = n_z
        self.m = m
        # Solve the program with

    def test_socp_derivatives_cpp(self):
        # Formulate and solve the same cone program with MOSEK.
        n_z = self.n_z
        m = self.m
        solver = MosekSolver()
        if not solver.enabled():
            solver = ScsSolver()
        prog = MathematicalProgram()
        z_mp = prog.NewContinuousVariables(n_z, "z")

        J_list = [J_i.value for J_i in self.J_cvx]
        e_list = [e_i.value for e_i in self.e_cvx]
        constraints = [
            prog.AddLorentzConeConstraint(J_i, e_i, z_mp)
            for J_i, e_i in zip(J_list, e_list)
        ]

        prog.AddLinearCost(self.b_cvx.value, z_mp)
        result = solver.Solve(prog, None, None)

        self.assertTrue(result.is_success())

        # Extract solutions.
        z_star = result.GetSolution(z_mp)
        lambda_star_list = [result.GetDualSolution(c) for c in constraints]

        # Compute derivatives.
        d_socp = SocpDerivatives(1e-2)
        G_list = [-J_i for J_i in J_list]
        d_socp.UpdateProblem(
            np.zeros((self.n_z, self.n_z)),
            self.b_cvx.value,
            G_list,
            e_list,
            z_star,
            lambda_star_list,
            1e-2,
            True,
        )

        # DzDb
        atol = 1e-4 if isinstance(solver, ScsSolver) else 1e-5
        np.testing.assert_allclose(self.DzDb, d_socp.get_DzDb(), atol=atol)

        # DzDe
        n_c = len(self.J_cvx)
        DzDe = d_socp.get_DzDe()
        for i in range(n_c):
            np.testing.assert_allclose(
                DzDe[:, i * m : i * m + m], self.DzDe[i], atol=1e-5
            )

        # DzDvecG
        DzDvecG_active, lambda_star_active_indices = d_socp.get_DzDvecG_active()
        n_active = len(lambda_star_active_indices)
        DzDvecG = []
        for i in range(n_active):
            DzDvecGi = np.zeros((n_z, n_z * m))

            for j in range(n_z):
                idx = j * n_z * m + i * m
                DzDvecGi[:, m * j : m * j + m] = DzDvecG_active[:, idx : idx + m]

            DzDvecG.append(DzDvecGi)

        for i in range(n_active):
            idx = lambda_star_active_indices[i]
            np.testing.assert_allclose(self.DzDvecJ[idx], -DzDvecG[i], atol=1e-4)
