
import time
import numpy as np
import matplotlib.pyplot as plt

from quasistatic_sim import *

q_sim = QuasistaticSimulator()

#%% Anitescu
q0 = np.array([0, r, -r * 1.1, r * 1.1, 0])

q = q0.copy()
q_sim.UpdateVisualizer(q)
print(q0)
input("start?")
# lambda_n_log = []
# lambda_f_log = []
q_log = []
beta_log = []

n_steps = 50
for i in range(n_steps):
    dr = np.min([0.001 * i, 0.02])
    q_a_cmd = np.array([-r * 1.1 + dr, r * 1.1 - dr, 0.002 * i])
    dq_a, dq_u, beta, result = q_sim.StepAnitescu(q, q_a_cmd)

    # Update q
    q += np.hstack([dq_u, dq_a])
    q_sim.UpdateVisualizer(q)

    # logging
    # lambda_n_log.append(lambda_n)
    # lambda_f_log.append(lambda_f)
    q_log.append(q.copy())
    beta_log.append(beta)

    time.sleep(h)

beta_log = np.array(beta_log)

#%%
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

# %%
fig, axes = plt.subplots(n_c, 1, figsize=(8, 9))
for i in range(n_c):
    axes[i].step(t_sim, lambda_n_log[:, i])
    axes[i].set_ylabel("lambda_n_{} [N]".format(i))
    axes[i].grid(True)
plt.show()

#%%
fig, axes = plt.subplots(n_c, 1, figsize=(8, 9))
for i in range(n_c):
    axes[i].step(t_sim, friction_log[:, i])
    axes[i].set_ylabel("friction_{} [N]".format(i))
    axes[i].grid(True)
plt.show()

#%%
fig, axes = plt.subplots(n_c, 1, figsize=(8, 9))
for i in range(n_c):
    axes[i].step(t_sim, phi_log[:, i])
    axes[i].set_ylabel("phi_{} [m]".format(i))
    axes[i].grid(True)
plt.show()
