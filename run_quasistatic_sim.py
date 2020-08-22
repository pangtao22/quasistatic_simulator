
import numpy as np
import matplotlib.pyplot as plt

from quasistatic_sim import QuasistaticSimulator, CalcPhi, r, h, n_c


q_sim = QuasistaticSimulator()

#%% old formulation as in paper, buggy.
q0 = np.array([0, r, -r - 0.001, r + 0.001, 0])
v_a_cmd = np.array([0.1, -0.1, 0.1])

q = q0.copy()
q_sim.UpdateVisualizer(q)
print(q0)
for i in range(100):
    dq_a, dq_u, lambda_n, lambda_f, result = q_sim.StepMiqp(q, v_a_cmd)
    q += np.hstack([dq_u, dq_a])
    q_sim.UpdateVisualizer(q)
    print(dq_u, dq_a, lambda_n)
    input("contune?")


#%% new "impedance robot" formulation, which I think is correct.
q0 = np.array([0, r, -r * 1.1, r * 1.1, 0])


q = q0.copy()
q_sim.UpdateVisualizer(q)
print(q0)
input("start?")
lambda_n_log = []
lambda_f_log = []
n_steps = 50
for i in range(n_steps):
    dr = np.min([0.001 * i, 0.02])
    q_a_cmd = np.array([-r * 1.1 + dr, r * 1.1 - dr, 0.002 * i])
    dq_a, dq_u, lambda_n, lambda_f, result = q_sim.StepLcp(q, q_a_cmd)
    lambda_n_log.append(lambda_n)
    lambda_f_log.append(lambda_f)
    q += np.hstack([dq_u, dq_a])
    q_sim.UpdateVisualizer(q)
    print(i, dq_u, dq_a, lambda_n)
    input("contune?")


# %% make some plots
time = np.arange(n_steps) * h
lambda_n_log = np.array(lambda_n_log)
lambda_f_log = np.array(lambda_f_log)
friction = np.zeros((n_steps, n_c))

for i in range(n_c):
    idx = i * 2
    friction[:, i] = lambda_f_log[:, idx] - lambda_f_log[:, idx+1]


#%%
fig, axes = plt.subplots(3, 1, figsize=(8, 9))
for i in range(n_c):
    axes[i].step(time, lambda_n_log[:, i])
    axes[i].set_ylabel("lambda_n_{} [N]".format(i))
    axes[i].grid(True)
plt.show()

#%%
fig, axes = plt.subplots(3, 1, figsize=(8, 9))
for i in range(n_c):
    axes[i].step(time, friction[:, i])
    axes[i].set_ylabel("friction_{} [N]".format(i))
    axes[i].grid(True)
plt.show()

#%%
plt.step(time, lambda_n_log[:, 0])
plt.grid(True)
plt.show()

