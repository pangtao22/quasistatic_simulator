import numpy as np
import cvxpy as cp

from qsim_cpp import SocpDerivatives

from pydrake.all import (MathematicalProgram, GurobiSolver, SolverOptions,
                         MosekSolver)
solver_mosek = MosekSolver()

#%%
n = 2
m = 2
J = cp.Parameter((m, n))
J.value = np.eye(n)

b = cp.Parameter(n)
b.value = np.array([1, 0])

e0 = cp.Parameter(m)
e0.value = np.zeros(m)

e1 = cp.Parameter(m)
e1.value = np.array([0, 2])

e2 = cp.Parameter(m)
e2.value = np.array([0, 1])

z = cp.Variable(n)
s0 = J @ z + e0
s1 = J @ z + e1
s2 = J @ z + e2

constraints = [cp.constraints.second_order.SOC(s0[0], s0[1:]),
               cp.constraints.second_order.SOC(s1[0], s1[1:]),
               cp.constraints.second_order.SOC(s2[0], s2[1:])]

prob = cp.Problem(cp.Minimize(b @ z), constraints)

prob.solve(requires_grad=True)

#%%
DzDb = np.zeros((n, n))
DzDe0 = np.zeros((n, m))
DzDe1 = np.zeros((n, m))
DzDe2 = np.zeros((n, m))

DzDJ = np.zeros((n, *J.shape))

for i in range(n):
    dv = np.zeros(n)
    dv[i] = 1
    z.gradient = dv
    prob.backward()

    DzDe0[i] = e0.gradient
    DzDe1[i] = e1.gradient
    DzDe2[i] = e2.gradient
    DzDb[i] = b.gradient
    DzDJ[i] = J.gradient


#%%
d_socp = SocpDerivatives(1e-2)

prog = MathematicalProgram()
z_mp = prog.NewContinuousVariables(n, "z")

J_list = [J.value, J.value, J.value]
e_list = [e0.value, e1.value, e2.value]
constraints = [prog.AddLorentzConeConstraint(J_i, e_i, z_mp)
               for J_i, e_i in zip(J_list, e_list)]

prog.AddLinearCost(b.value, z_mp)
result = solver_mosek.Solve(prog, None, None)

v_star = result.GetSolution(z_mp)
lambda_star_list = [result.GetDualSolution(c) for c in constraints]


#%%
G_list = [-J_i for J_i in J_list]
d_socp.UpdateProblem(
    np.zeros((n, n)), b.value, G_list, e_list, v_star, lambda_star_list,
    1e-2, False)


print(d_socp.get_DzDe())

