import numpy as np
import matplotlib.pyplot as plt
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver

solver = GurobiSolver()

#%%
"""
2D point mass supported by a plane under gravity (-y). 
"""
n_u = 2
M = np.eye(n_u) * 0  # kg in x, y.
tau_g = np.array([0, -10.0])  # Newtons.
v0 = np.array([0, 0])  # m/s
mu = 0.5  # coefficient of friction
k = 100.0  # N/m
h = 0.1  # s, time step.
d = 10  # damping
q_a_cmd_next = 0.0  # holding spring at origin.


def dynamics(q_u, q_a, v_u, q_a_cmd_next):
    prog = mp.MathematicalProgram()
    v_next = prog.NewContinuousVariables(3, "v")
    v_u_next = v_next[:n_u]
    v_a_next = v_next[n_u:]

    dq_a_cmd = q_a_cmd_next - q_a
    tau_h = np.hstack([M.dot(v_u) + tau_g * h, k * dq_a_cmd * h])

    Q = np.eye(n_u + 1)
    Q[:n_u, :n_u] = M
    Q[-1, -1] = k * h**2 + d * h
    prog.AddQuadraticCost(Q, -tau_h, v_next)

    phi = np.array([q_u[0] - q_a, q_u[1], q_u[1]])
    J = np.array([[1, 0, -1], [mu, 1, 0], [-mu, 1, 0]], dtype=float)
    constraints = prog.AddLinearConstraint(
        J, -phi / h, np.full_like(phi, np.inf), v_next
    )

    result = solver.Solve(prog)

    assert result.get_solution_result() == mp.SolutionResult.kSolutionFound
    v_next_values = result.GetSolution(v_next)
    q_u_next = q_u + v_next_values[:n_u] * h
    q_a_next = q_a + v_next_values[n_u] * h

    return q_u_next, q_a_next, v_next_values[:n_u]


#%%
q_a = 0.0
q_u = np.array([0.0, 0])
v_u = v0

L = int(0.5 / h)
q_u_log = [q_u]
v_u_log = [v_u]
q_a_log = [q_a]

t = np.arange(L + 1) * h

q_a_cmd = 0.2
for l in range(1, L + 1):
    q_u, q_a, v_u = dynamics(q_u, q_a, v_u, q_a_cmd)
    q_u_log.append(q_u)
    q_a_log.append(q_a)
    v_u_log.append(v_u)

q_a_log = np.array(q_a_log)
q_u_log = np.array(q_u_log)
v_u_log = np.array(v_u_log)

# compute kinetic energy
ke = 0.5 * (v_u_log.dot(M) * v_u_log).sum(axis=1)
pe = 0.5 * k * (q_a_cmd - q_a_log) ** 2


#%%
plt.figure()
plt.plot(t, q_u_log[:, 0], label="x")
plt.plot(t, q_u_log[:, 1], label="y")
plt.plot(t, q_a_log, label="q_a")
plt.legend()
plt.xlabel("t [s]")
plt.ylabel("q [m]")
plt.grid()
plt.show()

plt.figure()
plt.plot(t, ke, label="k_e")
plt.plot(t, pe, label="p_e")
plt.plot(t, ke + pe, label="total")
plt.legend()
plt.xlabel("t [s]")
plt.ylabel("energy [J]")
plt.grid()
plt.show()
