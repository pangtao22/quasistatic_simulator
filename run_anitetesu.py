
import time
import numpy as np
import matplotlib.pyplot as plt

from quasistatic_sim import *

q_sim = QuasistaticSimulator()



#%% Anitescu
q0 = np.array([0, r, -r, r, 0])
"""
If simulated with LCP:
At t = 0.10, gripper is commanded to touch (no penetration) the surface of 
the cylinder at t = 0.11. Hence contact force is 0 for t = 0.10 to 0.11.
At t = 0.11, penetration is commanded, leading to non-zero impulse from 
t = 0.11 to t = 0.12. 
"""

q = q0.copy()
q_sim.UpdateVisualizer(q)
print(q0)
input("start?")
q_log = [q0.copy()]
qa_cmd_log = []
beta_log = []
constraint_values_log = []

n_steps = 50
for i in range(n_steps):
    # dr = np.min([0.001 * i, 0.02])
    # q_a_cmd = np.array([-r * 1.1 + dr, r * 1.1 - dr, -0.002 * i])
    q_a_cmd = np.array([-r * 0.9, r * 0.9, np.max([-0.002 * i, -0.03])])
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



# %% plot normal force and distance
fig, axes = plt.subplots(n_c, 1, figsize=(8, 9))
for i in range(n_c):
    color = "red"
    axes[i].step(t_sim, lambda_n_log[:, i], where="post", color=color)
    axes[i].set_ylabel("c_n_{} [N]".format(i + 1), color=color)
    axes[i].tick_params(axis="y", labelcolor=color)
    axes[i].grid(True)

    ax2 = axes[i].twinx()
    color = "blue"
    ax2.step(t_sim, phi_log[:-1, i], 'o', where="post", color=color)
    ax2.set_ylabel("phi_{} [m]".format(i + 1), color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.grid(True)
plt.show()

#%% plot friction
fig, axes = plt.subplots(n_c + 1, 1, figsize=(8, 9))
for i in range(n_c):
    axes[i].step(t_sim, friction_log[:, i], where="post")
    axes[i].set_ylabel("friction_{} [N]".format(i + 1))
    axes[i].grid(True)
axes[3].step(t_sim, friction_log[:, 0] + friction_log[:, 1], where="post")
axes[3].set_ylabel("friction 1 + 2 [N]")
axes[3].grid(True)
plt.show()

#%% plot normal and tangent velocity
fig, axes = plt.subplots(n_c, 1, figsize=(8, 9))
for i in range(n_c):
    axes[i].step(t_sim, v_tangent[:, i * 2], where="post", label="t")
    axes[i].step(t_sim, v_normal[:, i], where="post", label="n")
    axes[i].set_ylabel("v{} [m/s]".format(i + 1))
    axes[i].grid(True)
    axes[i].legend()
plt.show()


#%% velocity
l = 11 # time step
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
