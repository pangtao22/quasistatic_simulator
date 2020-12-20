
import time
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 12})
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

from quasistatic_sim import *
from plotting import PlotForceDistance, PlotLeftFingerPosition

#%%
q_sim = QuasistaticSimulator()

#%% Anitescu

q = q0.copy()
q_sim.UpdateVisualizer(q)
print(q0)
# input("start?")
q_log = [q0.copy()]
qa_cmd_log = []
beta_log = []
constraint_values_log = []

input("start")
for i in range(n_steps):
    q_a_cmd = q_traj.value((i + 1) * h).squeeze()
    dq_a, dq_u, beta, constraint_values, result = q_sim.StepAnitescu(q, q_a_cmd)

    # Update q
    q += np.hstack([dq_u, dq_a])
    q_sim.UpdateVisualizer(q)

    # logging
    q_log.append(q.copy())
    beta_log.append(beta)
    qa_cmd_log.append(q_a_cmd)
    constraint_values_log.append(constraint_values)
    time.sleep(h * 10)


q_log = np.array(q_log)
qa_cmd_log = np.array(qa_cmd_log)
beta_log = np.array(beta_log)
constraint_values_log = np.array(constraint_values_log)

#%% compute data for plots
t_sim1 = np.arange(n_steps + 1) * h
t_sim = np.arange(n_steps) * h
phi_log = np.array([CalcPhi(q) for q in q_log])

lambda_n_log = np.zeros((n_steps, n_c))
friction_log = np.zeros((n_steps, n_c))

for l in range(n_steps):
    j_start = 0
    for i in range(n_c):
        lambda_n_log[l, i] = beta_log[l, j_start:j_start + n_d[i]].sum() / h
        friction_log[l, i] = beta_log[l, j_start] - beta_log[l, j_start + 1]
        friction_log[l, i] *= U[i, i] / h
        j_start += n_d[i]

Jf = np.hstack((Jf_u, Jf_a))
Jn = np.hstack((Jn_u, Jn_a))
dq = (q_log[1:] - q_log[:-1])
v_tangent = (Jf.dot(dq.T / h)).T
v_normal = (dq / h).dot(Jn.T)


#%%
PlotForceDistance(t_sim, phi_log, lambda_n_log, friction_log,
                  figsize=(6, 4),
                  save_name="contact_force_distance_qp.pdf")

PlotLeftFingerPosition(t_sim1, q_log, qa_cmd_log,
                       fig_size=(6, 3),
                       save_name="xy_cmd_vs_xy_true_qp.pdf")


#%% velocity
l = 11  # time step
i_c = 1  # contact idx
f = np.array([friction_log[l, i_c], lambda_n_log[l, i_c]])
v = np.array([v_tangent[l, i_c * 2], v_normal[l, i_c]])
v_length = np.linalg.norm(v)
f_length = np.linalg.norm(f)
if f_length > 0.1:
    f = f / f_length * 0.1
    f_length = 0.1
# plot boundaries
length = np.max((f_length, v_length))
x = np.linspace(-length, length, 51)
mu = U[0, 0]
y_v1 = -mu * x - phi_log[l, i_c] / h
y_v2 = mu * x - phi_log[l, i_c] / h
y_f1 = mu * x[25:]
y_f2 = -mu * x[25:]
plt.plot(x, y_v1, '--', color="blue", label="(1)", zorder=2)
plt.plot(x, y_v2, '--', color="red", label="(2)", zorder=2)
plt.plot(y_f1, x[25:], '--', color="blue", zorder=2)
plt.plot(y_f2, x[25:], '--', color="red", zorder=2)

plt.arrow(0, 0, v[0], v[1], length_includes_head=True, edgecolor=None,
          color="black", lw=3, zorder=0)
plt.arrow(0, 0, f[0], f[1], length_includes_head=True, edgecolor=None,
          color="green", lw=3, zorder=0)

plt.grid(True)
plt.axis("equal")
plt.xlabel("tangent [m/s]")
plt.ylabel("normal [m/s]")
plt.title("t = {}".format(l * h))
plt.legend()
plt.show()

#%% forces for free-body diagrams
c_n_log = Jn.T.dot(E.T.dot(beta_log.T)) / h
c_f_log = Jf.T.dot(mu * beta_log.T) / h
c_n_log = c_n_log.T
c_f_log = c_f_log.T
spring_force = Kq_a.dot(qa_cmd_log.T - q_log[1:, n_u:].T)
spring_force = spring_force.T

residual_u = c_n_log[:, :n_u] + c_f_log[:, :n_u] + tau_ext
residual_a = c_n_log[:, n_u:] + c_f_log[:, n_u:] + spring_force


#%%
print("c_n", c_n_log[l])
print("c_f", c_f_log[l])
print("spring_force", spring_force[l])
print("qa_cmd_l", qa_cmd_log[l])
print("q_l", q_log[l])
print("q_(l+1)", q_log[l+1])
