import warnings

import meshcat

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver

# from problem_definition_pinch import *
from problem_definition_graze import *
from meshcat_camera_utils import SetOrthographicCameraXY


class QuasistaticSimulator:
    def __init__(self, visualize=False, is_quasi_dynamic=False):
        self.solver = GurobiSolver()
        assert self.solver.available()

        self.is_quasi_dynamic = is_quasi_dynamic

        self.vis = None
        if visualize:
            self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
            self.vis["cylinder"].set_object(
                meshcat.geometry.Cylinder(height=0.1, radius=r),
                meshcat.geometry.MeshLambertMaterial(
                    color=0xffffff, opacity=0.5))
            self.vis["cylinder"].set_transform(
                meshcat.transformations.euler_matrix(np.pi / 2, 0, 0))
            self.finger_thickness = 0.02
            self.vis["left_finger"].set_object(
                meshcat.geometry.Box([self.finger_thickness, 0.1, 0.2]),
                meshcat.geometry.MeshLambertMaterial(
                    color=0xff0000, opacity=1.0))
            self.vis["right_finger"].set_object(
                meshcat.geometry.Box([self.finger_thickness, 0.1, 0.2]),
                meshcat.geometry.MeshLambertMaterial(
                    color=0xff0000, opacity=1.0))
            self.vis["support"].set_object(
                meshcat.geometry.Box([1, 1, 0.1]),
                meshcat.geometry.MeshLambertMaterial(
                    color=0x00ff00, opacity=1.0))
            self.vis["support"].set_transform(
                meshcat.transformations.translation_matrix([0, -0.5, 0]))

            SetOrthographicCameraXY(self.vis)

    def init_program(self, phi_l):
        prog = mp.MathematicalProgram()

        # Declare decision variables.
        dq_u = prog.NewContinuousVariables(n_u, "dq_u")
        dq_a = prog.NewContinuousVariables(n_a, "dq_a")
        P_n = prog.NewContinuousVariables(n_c, "P_n")
        P_f = prog.NewContinuousVariables(n_f, "P_f")
        Gamma = prog.NewContinuousVariables(n_c, "Gamma")

        # Bounding box constraints for decision variables.
        prog.AddBoundingBoxConstraint(-dq_max, dq_max, dq_u)
        prog.AddBoundingBoxConstraint(-dq_max, dq_max, dq_a)
        prog.AddBoundingBoxConstraint(0, impulse_max, P_n)
        prog.AddBoundingBoxConstraint(0, impulse_max, P_f)
        prog.AddBoundingBoxConstraint(0, np.inf, Gamma)

        phi_l1 = Jn_u.dot(dq_u) + Jn_a.dot(dq_a) + phi_l
        rho = Jf_u.dot(dq_u) + Jf_a.dot(dq_a) + E.dot(Gamma)
        s = U.dot(P_n) - E.T.dot(P_f)

        for phi_l1_i in phi_l1:
            prog.AddLinearConstraint(phi_l1_i >= 0)

        for rho_i in rho:
            prog.AddLinearConstraint(rho_i >= 0)

        for s_i in s:
            prog.AddLinearConstraint(s_i >= 0)

        # Force balance constraint
        tau_sum = Jn_u.T.dot(P_n) + Jf_u.T.dot(P_f) + P_ext
        rhs = np.zeros(n_u)
        if self.is_quasi_dynamic:
            rhs = M_u @ dq_u / h
        for rhs_i, tau_sum_i in zip(rhs, tau_sum):
            prog.AddLinearConstraint(tau_sum_i == rhs_i)

        return prog, dq_u, dq_a, P_n, P_f, Gamma, phi_l1, rho, s, tau_sum

    @staticmethod
    def find_big_M(phi_l):
        M_phi = np.zeros(n_c)
        # Jn_u * dq_u + Jn_a * dq_a + phi_l >= 0
        for i in range(n_c):
            M_phi[i] += phi_l[i]
            for Ji in Jn_u[i]:
                M_phi[i] += dq_max * abs(Ji)
            for Ji in Jn_a[i]:
                M_phi[i] += dq_max * abs(Ji)

        M_phi = np.maximum(M_phi, impulse_max)

        M_rho = np.zeros(n_f)
        M_gamma = np.zeros(n_c)
        j_start = 0
        for i in range(n_c):
            M_i = np.zeros(n_d[i])
            for j in range(n_d[i]):
                idx = j_start + j
                for Ji in Jf_u[idx]:
                    M_i[j] += dq_max * abs(Ji)
                for Ji in Jf_a[idx]:
                    M_i[j] += dq_max * abs(Ji)
            M_gamma[i] = np.max(M_i) * 2
            M_rho[j_start:j_start + n_d[i]] = M_gamma[i]
            j_start += n_d[i]
        M_rho = np.maximum(M_rho, impulse_max)

        M_s = np.maximum(U.diagonal() * impulse_max, M_gamma)

        return M_phi, M_rho, M_s

    def step_miqp(self, q, v_a_cmd):
        phi_l = calc_phi(q)
        dq_a_cmd = v_a_cmd * h

        (prog, dq_u, dq_a, P_n, P_f, Gamma, phi_l1, rho, s,
         tau_sum) = self.init_program(phi_l)
        M_phi, M_rho, M_s = self.find_big_M(phi_l)

        z_phi = prog.NewBinaryVariables(n_c, "z_phi")
        z_rho = prog.NewBinaryVariables(n_f, "z_rho")
        z_s = prog.NewBinaryVariables(n_c, "z_s")

        for i in range(n_c):
            prog.AddLinearConstraint(phi_l1[i] <= M_phi[i] * z_phi[i])
            prog.AddLinearConstraint(P_n[i] <= M_phi[i] * (1 - z_phi[i]))

            prog.AddLinearConstraint(s[i] <= M_s[i] * z_s[i])
            prog.AddLinearConstraint(Gamma[i] <= M_s[i] * (1 - z_s[i]))

        for i in range(n_f):
            prog.AddLinearConstraint(rho[i] <= M_rho[i] * z_rho[i])
            prog.AddLinearConstraint(P_f[i] <= M_rho[i] * (1 - z_rho[i]))

        phi_bar = Jn_u.dot(dq_u) + Jn_a.dot(dq_a_cmd) + phi_l
        for i in range(n_c):
            prog.AddLinearConstraint(P_n[i] >= - 1e4 * h * phi_bar[i])

        prog.AddQuadraticCost(((dq_a - dq_a_cmd) ** 2).sum())

        result = self.solver.Solve(prog, None, None)

        dq_a = result.GetSolution(dq_a)
        dq_u = result.GetSolution(dq_u)
        lambda_n = result.GetSolution(P_n) / h
        lambda_f = result.GetSolution(P_f) / h

        return dq_a, dq_u, lambda_n, lambda_f, result

    def step_lcp(self, q, q_a_cmd):
        """
        The problem is an LCP although it is solved as an MIQP due to the
        lack of the PATH solver.
        :param q:
        :param v_a_cmd:
        :return:
        """
        phi_l = calc_phi(q)
        dq_a_cmd = q_a_cmd - q[n_u:]

        (prog, dq_u, dq_a, P_n, P_f, Gamma, phi_l1, rho, s,
         tau_sum) = self.init_program(phi_l)
        M_phi, M_rho, M_s = self.find_big_M(phi_l)

        z_phi = prog.NewBinaryVariables(n_c, "z_phi")
        z_rho = prog.NewBinaryVariables(n_f, "z_rho")
        z_s = prog.NewBinaryVariables(n_c, "z_s")

        for i in range(n_c):
            prog.AddLinearConstraint(phi_l1[i] <= M_phi[i] * z_phi[i])
            prog.AddLinearConstraint(P_n[i] <= M_phi[i] * (1 - z_phi[i]))

            prog.AddLinearConstraint(s[i] <= M_s[i] * z_s[i])
            prog.AddLinearConstraint(Gamma[i] <= M_s[i] * (1 - z_s[i]))

        for i in range(n_f):
            prog.AddLinearConstraint(rho[i] <= M_rho[i] * z_rho[i])
            prog.AddLinearConstraint(P_f[i] <= M_rho[i] * (1 - z_rho[i]))

        # force balance for robot.
        tau_a = Jn_a.T.dot(P_n) + Jf_a.T.dot(P_f) + \
                Kq_a.dot(dq_a_cmd - dq_a) * h

        for i in range(n_a):
            prog.AddLinearConstraint(tau_a[i] == 0)

        result = self.solver.Solve(prog, None, None)

        dq_a = result.GetSolution(dq_a)
        dq_u = result.GetSolution(dq_u)
        lambda_n = result.GetSolution(P_n) / h
        lambda_f = result.GetSolution(P_f) / h

        return dq_a, dq_u, lambda_n, lambda_f, result

    def step_anitescu(self, q, q_a_cmd):
        phi_l = calc_phi(q)
        dq_a_cmd = q_a_cmd - q[n_u:]

        prog = mp.MathematicalProgram()
        v_u = prog.NewContinuousVariables(n_u, "dq_u")
        v_a = prog.NewContinuousVariables(n_a, "dq_a")

        prog.AddQuadraticCost(Kq_a * h**2, -Kq_a.dot(dq_a_cmd) * h, v_a)
        prog.AddLinearCost(-P_ext, 0, v_u)
        if self.is_quasi_dynamic:
            prog.AddQuadraticCost(M_u, np.zeros(n_u), v_u)

        Jn = np.hstack([Jn_u, Jn_a])
        Jf = np.hstack([Jf_u, Jf_a])
        J = np.zeros_like(Jf)
        phi_constraints = np.zeros(n_f)

        j_start = 0
        for i in range(n_c):
            for j in range(n_d[i]):
                idx = j_start + j
                J[idx] = Jn[i] + U[i, i] * Jf[idx]
                phi_constraints[idx] = phi_l[i]
            j_start += n_d[i]

        v = np.hstack([v_u, v_a])
        constraints = prog.AddLinearConstraint(
            J, -phi_constraints / h, np.full_like(phi_constraints, np.inf), v)

        result = self.solver.Solve(prog, None, None)
        beta = result.GetDualSolution(constraints)
        beta = np.array(beta).squeeze()
        dq_a = result.GetSolution(v_a) * h
        dq_u = result.GetSolution(v_u) * h
        constraint_values = phi_constraints + result.EvalBinding(constraints)

        return dq_a, dq_u, beta, constraint_values, result

    def update_visualizer(self, q):
        if not self.vis:
            warnings.warn("Simulator is not set up to visualize.")
            return

        qu = q[:n_u]
        qa = q[n_u:]
        xc, yc = qu
        xl, xr, yg = qa
        self.vis["left_finger"].set_transform(
            meshcat.transformations.translation_matrix(
                [xl - self.finger_thickness / 2, yg + 0.1, 0]))
        self.vis["right_finger"].set_transform(
            meshcat.transformations.translation_matrix(
                [xr + self.finger_thickness / 2, yg + 0.1, 0]))
        X_WC = meshcat.transformations.euler_matrix(np.pi / 2, 0, 0)
        X_WC[0:3, 3] = [xc, yc, 0]
        self.vis["cylinder"].set_transform(X_WC)


