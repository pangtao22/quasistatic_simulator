import time
import numpy as np
import matplotlib.pyplot as plt

from quasistatic_sim import QuasistaticSimulator, CalcPhi, r, h, n_c

q_sim = QuasistaticSimulator()

#%% new "impedance robot" formulation, which I think is correct.
q0 = np.array([0, r, -r * 1.1, r * 1.1, 0])

q = q0.copy()
q_sim.UpdateVisualizer(q)
print(q0)
input("start?")
lambda_n_log = []
lambda_f_log = []
q_log = []
n_steps = 50
for i in range(n_steps):
    dr = np.min([0.001 * i, 0.02])
    q_a_cmd = np.array([-r * 1.1 + dr, r * 1.1 - dr, 0.002 * i])
    dq_a, dq_u, lambda_n, lambda_f, result = q_sim.StepLcp(q, q_a_cmd)

    # Update q.
    q += np.hstack([dq_u, dq_a])
    q_sim.UpdateVisualizer(q)

    # logging.
    lambda_n_log.append(lambda_n)
    lambda_f_log.append(lambda_f)
    q_log.append(q.copy())

    time.sleep(h)


# %% make some plots
t_sim = np.arange(n_steps) * h
lambda_n_log = np.array(lambda_n_log)
lambda_f_log = np.array(lambda_f_log)
friction_log = np.zeros((n_steps, n_c))
phi_log = np.array([CalcPhi(q) for q in q_log])

for i in range(n_c):
    idx = i * 2
    friction_log[:, i] = lambda_f_log[:, idx] - lambda_f_log[:, idx + 1]


#%%
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
