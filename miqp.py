import numpy as np

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver

#%%
solver = GurobiSolver()
assert solver.available()
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

E = np.zeros((n_f, n_c))
i_start = 0
for i in range(n_c):
    i_end = i_start + n_d[i]
    E[i_start: i_end, i] = 1
    i_start += n_d[i]

U = np.eye(n_c) * 0.5


#%% define some functions
h = 0.01  # simulation time step
dq_max = 1 * h  # 1m/s
impulse_max = 50 * h  # 50N
tau_ext = np.array([0., -10])
P_ext = tau_ext * h
phi_l = np.array([0., 0., 0])


def InitProgram(phi_l):
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
    for tau_sum_i in tau_sum:
        prog.AddLinearConstraint(tau_sum_i == 0)

    return prog, dq_u, dq_a, P_n, P_f, Gamma, phi_l1, rho, s, tau_sum


# %% find big M
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
i_start = 0
for i in range(n_c):
    M_i = np.zeros(n_d[i])
    for j in range(n_d[i]):
        idx = i_start + j
        print(idx)
        for Ji in Jf_u[idx]:
            M_i[j] += dq_max * abs(Ji)
        for Ji in Jf_a[idx]:
            M_i[j] += dq_max * abs(Ji)
    print(M_i)
    M_gamma[i] = np.max(M_i) * 2
    M_rho[i_start:i_start + n_d[i]] = M_gamma[i]
    i_start += n_d[i]
M_rho = np.maximum(M_rho, impulse_max)

M_s = np.maximum(U.diagonal() * impulse_max, M_gamma)



#%% construct program to solve for forces and displacements
M_value = 1
v_a_cmd = np.array([0.1, -0.1, 0.1])
dq_a_cmd = v_a_cmd * h

prog, dq_u, dq_a, P_n, P_f, Gamma, phi_l1, rho, s, tau_sum = InitProgram(phi_l)

z_phi = prog.NewBinaryVariables(n_c, "z_phi")
z_rho = prog.NewBinaryVariables(n_f, "z_rho")
z_s = prog.NewBinaryVariables(n_c, "z_s")

# M_phi[:] = 1
# M_s[:] = 1
# M_rho[:] = 1

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

prog.AddQuadraticCost(((dq_a - dq_a_cmd)**2).sum())
result = solver.Solve(prog, None, None)
print(result.get_solution_result())
print("dq_a", result.GetSolution(dq_a))
print("dq_u", result.GetSolution(dq_u))
print("lambda_n", result.GetSolution(P_n) / h)
print("lambda_f", result.GetSolution(P_f) / h)
