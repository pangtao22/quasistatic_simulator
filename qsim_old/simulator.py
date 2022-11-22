import warnings
from typing import Dict

import numpy as np
import meshcat

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver

#
# # from problem_definition_pinch import *
# from q_sim_old.problem_definition_graze import *
from qsim_old.meshcat_camera_utils import SetOrthographicCameraXY


def calc_E(n_d, n_c):
    E = np.zeros((n_d.sum(), n_c))
    i_start = 0
    for i in range(n_c):
        i_end = i_start + n_d[i]
        E[i_start:i_end, i] = 1
        i_start += n_d[i]
    return E


class QuasistaticSimulator:
    def __init__(
        self, problem_definition: Dict, visualize=False, is_quasi_dynamic=False
    ):
        self.solver = GurobiSolver()
        assert self.solver.available()

        self.n_u = problem_definition["n_u"]
        self.n_a = problem_definition["n_a"]
        self.n_c = problem_definition["n_c"]
        self.n_d = problem_definition["n_d"]
        self.n_f = problem_definition["n_f"]
        self.Jn_u = problem_definition["Jn_u"]
        self.Jn_a = problem_definition["Jn_a"]
        self.Jf_u = problem_definition["Jf_u"]
        self.Jf_a = problem_definition["Jf_a"]
        self.U = problem_definition["U"]
        self.M_u = problem_definition["M_u"]
        self.tau_ext = problem_definition["tau_ext"]
        self.Kq_a = problem_definition["Kq_a"]
        self.calc_phi = problem_definition["calc_phi"]
        self.h = problem_definition["h"]
        self.dq_max = problem_definition["dq_max"]
        self.impulse_max = problem_definition["impulse_max"]
        self.P_ext = problem_definition["P_ext"]
        self.E = calc_E(self.n_d, self.n_c)

        self.is_quasi_dynamic = is_quasi_dynamic

        self.vis = None
        if visualize:
            # only makes sense for the 2d gripper ball pinch problem.
            r = problem_definition["r"]
            self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
            self.vis["cylinder"].set_object(
                meshcat.geometry.Cylinder(height=0.1, radius=r),
                meshcat.geometry.MeshLambertMaterial(
                    color=0xFFFFFF, opacity=0.5
                ),
            )
            self.vis["cylinder"].set_transform(
                meshcat.transformations.euler_matrix(np.pi / 2, 0, 0)
            )
            self.finger_thickness = 0.02
            self.vis["left_finger"].set_object(
                meshcat.geometry.Box([self.finger_thickness, 0.1, 0.2]),
                meshcat.geometry.MeshLambertMaterial(
                    color=0xFF0000, opacity=1.0
                ),
            )
            self.vis["right_finger"].set_object(
                meshcat.geometry.Box([self.finger_thickness, 0.1, 0.2]),
                meshcat.geometry.MeshLambertMaterial(
                    color=0xFF0000, opacity=1.0
                ),
            )
            self.vis["support"].set_object(
                meshcat.geometry.Box([1, 1, 0.1]),
                meshcat.geometry.MeshLambertMaterial(
                    color=0x00FF00, opacity=1.0
                ),
            )
            self.vis["support"].set_transform(
                meshcat.transformations.translation_matrix([0, -0.5, 0])
            )

            SetOrthographicCameraXY(self.vis)

    def init_program(self, phi_l):
        prog = mp.MathematicalProgram()

        # Declare decision variables.
        dq_u = prog.NewContinuousVariables(self.n_u, "dq_u")
        dq_a = prog.NewContinuousVariables(self.n_a, "dq_a")
        P_n = prog.NewContinuousVariables(self.n_c, "P_n")
        P_f = prog.NewContinuousVariables(self.n_f, "P_f")
        Gamma = prog.NewContinuousVariables(self.n_c, "Gamma")

        # Bounding box constraints for decision variables.
        prog.AddBoundingBoxConstraint(-self.dq_max, self.dq_max, dq_u)
        prog.AddBoundingBoxConstraint(-self.dq_max, self.dq_max, dq_a)
        prog.AddBoundingBoxConstraint(0, self.impulse_max, P_n)
        prog.AddBoundingBoxConstraint(0, self.impulse_max, P_f)
        prog.AddBoundingBoxConstraint(0, np.inf, Gamma)

        phi_l1 = self.Jn_u.dot(dq_u) + self.Jn_a.dot(dq_a) + phi_l
        rho = self.Jf_u.dot(dq_u) + self.Jf_a.dot(dq_a) + self.E.dot(Gamma)
        s = self.U.dot(P_n) - self.E.T.dot(P_f)

        for phi_l1_i in phi_l1:
            prog.AddLinearConstraint(phi_l1_i >= 0)

        for rho_i in rho:
            prog.AddLinearConstraint(rho_i >= 0)

        for s_i in s:
            prog.AddLinearConstraint(s_i >= 0)

        # Force balance constraint
        tau_sum = self.Jn_u.T.dot(P_n) + self.Jf_u.T.dot(P_f) + self.P_ext
        rhs = np.zeros(self.n_u)
        if self.is_quasi_dynamic:
            rhs = self.M_u @ dq_u / self.h
        for rhs_i, tau_sum_i in zip(rhs, tau_sum):
            prog.AddLinearConstraint(tau_sum_i == rhs_i)

        return prog, dq_u, dq_a, P_n, P_f, Gamma, phi_l1, rho, s, tau_sum

    def find_big_M(self, phi_l):
        M_phi = np.zeros(self.n_c)
        # Jn_u * dq_u + Jn_a * dq_a + phi_l >= 0
        for i in range(self.n_c):
            M_phi[i] += phi_l[i]
            for Ji in self.Jn_u[i]:
                M_phi[i] += self.dq_max * abs(Ji)
            for Ji in self.Jn_a[i]:
                M_phi[i] += self.dq_max * abs(Ji)

        M_phi = np.maximum(M_phi, self.impulse_max)

        M_rho = np.zeros(self.n_f)
        M_gamma = np.zeros(self.n_c)
        j_start = 0
        for i in range(self.n_c):
            M_i = np.zeros(self.n_d[i])
            for j in range(self.n_d[i]):
                idx = j_start + j
                for Ji in self.Jf_u[idx]:
                    M_i[j] += self.dq_max * abs(Ji)
                for Ji in self.Jf_a[idx]:
                    M_i[j] += self.dq_max * abs(Ji)
            M_gamma[i] = np.max(M_i) * 2
            M_rho[j_start : j_start + self.n_d[i]] = M_gamma[i]
            j_start += self.n_d[i]
        M_rho = np.maximum(M_rho, self.impulse_max)

        M_s = np.maximum(self.U.diagonal() * self.impulse_max, M_gamma)

        return M_phi, M_rho, M_s

    def step_miqp(self, q, v_a_cmd):
        """
        This seems to be based on my IROS2018 paper. It does not seem to
        generate sensible results right now... don't use!
        """
        phi_l = self.calc_phi(q)
        dq_a_cmd = v_a_cmd * self.h

        (
            prog,
            dq_u,
            dq_a,
            P_n,
            P_f,
            Gamma,
            phi_l1,
            rho,
            s,
            tau_sum,
        ) = self.init_program(phi_l)
        M_phi, M_rho, M_s = self.find_big_M(phi_l)

        z_phi = prog.NewBinaryVariables(self.n_c, "z_phi")
        z_rho = prog.NewBinaryVariables(self.n_f, "z_rho")
        z_s = prog.NewBinaryVariables(self.n_c, "z_s")

        for i in range(self.n_c):
            prog.AddLinearConstraint(phi_l1[i] <= M_phi[i] * z_phi[i])
            prog.AddLinearConstraint(P_n[i] <= M_phi[i] * (1 - z_phi[i]))

            prog.AddLinearConstraint(s[i] <= M_s[i] * z_s[i])
            prog.AddLinearConstraint(Gamma[i] <= M_s[i] * (1 - z_s[i]))

        for i in range(self.n_f):
            prog.AddLinearConstraint(rho[i] <= M_rho[i] * z_rho[i])
            prog.AddLinearConstraint(P_f[i] <= M_rho[i] * (1 - z_rho[i]))

        # TODO: what is this constraint? What is the magic number 1e4?
        phi_bar = self.Jn_u.dot(dq_u) + self.Jn_a.dot(dq_a_cmd) + phi_l
        for i in range(self.n_c):
            prog.AddLinearConstraint(P_n[i] >= -1e4 * self.h * phi_bar[i])

        prog.AddQuadraticCost(((dq_a - dq_a_cmd) ** 2).sum())

        result = self.solver.Solve(prog, None, None)

        dq_a = result.GetSolution(dq_a)
        dq_u = result.GetSolution(dq_u)
        lambda_n = result.GetSolution(P_n) / self.h
        lambda_f = result.GetSolution(P_f) / self.h

        return dq_a, dq_u, lambda_n, lambda_f, result

    def step_lcp(self, q, q_a_cmd):
        """
        The problem is an LCP although it is solved as an MIQP due to the
        lack of the PATH solver.
        :param q:
        :param v_a_cmd:
        :return:
        """
        phi_l = self.calc_phi(q)
        dq_a_cmd = q_a_cmd - q[self.n_u :]

        (
            prog,
            dq_u,
            dq_a,
            P_n,
            P_f,
            Gamma,
            phi_l1,
            rho,
            s,
            tau_sum,
        ) = self.init_program(phi_l)
        M_phi, M_rho, M_s = self.find_big_M(phi_l)

        z_phi = prog.NewBinaryVariables(self.n_c, "z_phi")
        z_rho = prog.NewBinaryVariables(self.n_f, "z_rho")
        z_s = prog.NewBinaryVariables(self.n_c, "z_s")

        for i in range(self.n_c):
            prog.AddLinearConstraint(phi_l1[i] <= M_phi[i] * z_phi[i])
            prog.AddLinearConstraint(P_n[i] <= M_phi[i] * (1 - z_phi[i]))

            prog.AddLinearConstraint(s[i] <= M_s[i] * z_s[i])
            prog.AddLinearConstraint(Gamma[i] <= M_s[i] * (1 - z_s[i]))

        for i in range(self.n_f):
            prog.AddLinearConstraint(rho[i] <= M_rho[i] * z_rho[i])
            prog.AddLinearConstraint(P_f[i] <= M_rho[i] * (1 - z_rho[i]))

        # force balance for robot.
        tau_a = (
            self.Jn_a.T.dot(P_n)
            + self.Jf_a.T.dot(P_f)
            + self.Kq_a.dot(dq_a_cmd - dq_a) * self.h
        )

        for i in range(self.n_a):
            prog.AddLinearConstraint(tau_a[i] == 0)

        result = self.solver.Solve(prog, None, None)

        dq_a = result.GetSolution(dq_a)
        dq_u = result.GetSolution(dq_u)
        lambda_n = result.GetSolution(P_n) / self.h
        lambda_f = result.GetSolution(P_f) / self.h

        return dq_a, dq_u, lambda_n, lambda_f, result

    def step_anitescu(self, q, q_a_cmd):
        phi_l = self.calc_phi(q)
        dq_a_cmd = q_a_cmd - q[self.n_u :]

        prog = mp.MathematicalProgram()
        v_u = prog.NewContinuousVariables(self.n_u, "dq_u")
        v_a = prog.NewContinuousVariables(self.n_a, "dq_a")

        prog.AddQuadraticCost(
            self.Kq_a * self.h**2, -self.Kq_a.dot(dq_a_cmd) * self.h, v_a
        )
        prog.AddLinearCost(-self.P_ext, 0, v_u)
        if self.is_quasi_dynamic:
            prog.AddQuadraticCost(self.M_u, np.zeros(self.n_u), v_u)

        Jn = np.hstack([self.Jn_u, self.Jn_a])
        Jf = np.hstack([self.Jf_u, self.Jf_a])
        J = np.zeros_like(Jf)
        phi_constraints = np.zeros(self.n_f)

        j_start = 0
        for i in range(self.n_c):
            for j in range(self.n_d[i]):
                idx = j_start + j
                J[idx] = Jn[i] + self.U[i, i] * Jf[idx]
                phi_constraints[idx] = phi_l[i]
            j_start += self.n_d[i]

        v = np.hstack([v_u, v_a])
        constraints = prog.AddLinearConstraint(
            J,
            -phi_constraints / self.h,
            np.full_like(phi_constraints, np.inf),
            v,
        )

        result = self.solver.Solve(prog, None, None)
        beta = result.GetDualSolution(constraints)
        beta = np.array(beta).squeeze()
        dq_a = result.GetSolution(v_a) * self.h
        dq_u = result.GetSolution(v_u) * self.h
        constraint_values = phi_constraints + result.EvalBinding(constraints)

        return dq_a, dq_u, beta, constraint_values, result

    def update_visualizer(self, q):
        if not self.vis:
            warnings.warn("Simulator is not set up to visualize.")
            return

        qu = q[: self.n_u]
        qa = q[self.n_u :]
        xc, yc = qu
        xl, xr, yg = qa
        self.vis["left_finger"].set_transform(
            meshcat.transformations.translation_matrix(
                [xl - self.finger_thickness / 2, yg + 0.1, 0]
            )
        )
        self.vis["right_finger"].set_transform(
            meshcat.transformations.translation_matrix(
                [xr + self.finger_thickness / 2, yg + 0.1, 0]
            )
        )
        X_WC = meshcat.transformations.euler_matrix(np.pi / 2, 0, 0)
        X_WC[0:3, 3] = [xc, yc, 0]
        self.vis["cylinder"].set_transform(X_WC)
