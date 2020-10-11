import time
import numpy as np
import matplotlib.pyplot as plt

from quasistatic_sim import *
#%%
q_sim = QuasistaticSimulator()

#%% new "impedance robot" formulation, which I think is correct.
# q0 = np.array([0, r, -r * 1.1, r * 1.1, 0])
q0 = np.array([0, r, -r, r, 0])

q = q0.copy()
q_sim.UpdateVisualizer(q)
print(q0)
input("start?")
lambda_n_log = []
lambda_f_log = []
q_log = [q0.copy()]
qa_cmd_log = []
n_steps = 50
for i in range(n_steps):
    # dr = np.min([0.001 * i, 0.02])
    # q_a_cmd = np.array([-r * 1.1 + dr, r * 1.1 - dr, -0.002 * i])
    q_a_cmd = np.array([-r * 0.9, r * 0.9, np.max([-0.002 * i, -0.03])])
    dq_a, dq_u, lambda_n, lambda_f, result = q_sim.StepLcp(q, q_a_cmd)

    # Update q.
    q += np.hstack([dq_u, dq_a])
    q_sim.UpdateVisualizer(q)

    # logging.
    lambda_n_log.append(lambda_n)
    lambda_f_log.append(lambda_f)
    q_log.append(q.copy())
    qa_cmd_log.append(q_a_cmd)

    time.sleep(h * 10)

q_log = np.array(q_log)
qa_cmd_log = np.array(qa_cmd_log)
lambda_n_log = np.array(lambda_n_log)
lambda_f_log = np.array(lambda_f_log)

# %% compute data for plots
"""
lambda_n_log[i] is the impulse over [h*i, h*(i+1)]  
"""
t_sim = np.arange(n_steps) * h
friction_log = np.zeros((n_steps, n_c))
contact_velocity_log = np.zeros((n_c, n_steps))
phi_log = np.array([CalcPhi(q) for q in q_log])

for i in range(n_c):
    idx = i * 2
    friction_log[:, i] = lambda_f_log[:, idx] - lambda_f_log[:, idx + 1]


Jf = np.hstack((Jf_u, Jf_a))
Jn = np.hstack((Jn_u, Jn_a))
dq = (q_log[1:] - q_log[:-1])
v_tangent = (Jf.dot(dq.T / h)).T
v_normal = (dq / h).dot(Jn.T)



#%% plot normal force and distance
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
fig, axes = plt.subplots(n_c, 1, figsize=(8, 9))
for i in range(n_c):
    axes[i].step(t_sim, friction_log[:, i], where="post")
    axes[i].set_ylabel("friction_{} [N]".format(i + 1))
    axes[i].grid(True)
plt.show()


