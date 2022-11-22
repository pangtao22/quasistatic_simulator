import numpy as np
import matplotlib.pyplot as plt
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver

solver = GurobiSolver()

#%%
"""
1D point mass collides with a spring-damper system. No gravity.
Even without damping (d=0), damping is provided as a side effect of 
implicit-Euler integration, which seems to grow with the time step h. This 
damping seems substantial when h=0.1. 
"""
m = 1.0  # kg
v0 = -0.1  # m/s
k = 100.0  # N/m
h = 0.1  # s, time step.
d = 1  # damping
q_a_cmd_next = 0.0  # holding spring at origin.


def dynamics(q_u, q_a, v_u):
    prog = mp.MathematicalProgram()
    v_next = prog.NewContinuousVariables(2, "v")
    # v_u_next = v_next[0]
    # v_a_next = v_next[1]

    dq_a_cmd = q_a_cmd_next - q_a

    tau_h = np.array([m * v_u, k * dq_a_cmd * h])
    Q = np.diag([m, k * h**2 + d * h])
    phi = q_u - q_a
    prog.AddQuadraticCost(Q, -tau_h, v_next)
    prog.AddLinearConstraint(phi / h + np.array([1, -1]).dot(v_next) >= 0)

    result = solver.Solve(prog)

    print(result.get_solution_result())
    v_next_values = result.GetSolution(v_next)
    q_u_next = q_u + v_next_values[0] * h
    q_a_next = q_a + v_next_values[1] * h

    return q_u_next, q_a_next, v_next_values[0]


#%%
q_a = 0.0
q_u = 0.0
v_u = v0

L = int(1 / h)
q_u_log = [q_u]
v_u_log = [v_u]
q_a_log = [q_a]

t = np.arange(L + 1) * h

for _ in range(L):
    q_u, q_a, v_u = dynamics(q_u, q_a, v_u)
    q_u_log.append(q_u)
    q_a_log.append(q_a)
    v_u_log.append(v_u)

q_a_log = np.array(q_a_log)
q_u_log = np.array(q_u_log)
v_u_log = np.array(v_u_log)

# compute kinetic energy
ke = 0.5 * m * v_u_log**2
pe = 0.5 * k * q_a_log**2


#%%
plt.figure()
plt.plot(t, q_u_log, label="q_u")
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
