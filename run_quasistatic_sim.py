import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

from quasistatic_sim import *
from plotting import PlotForceDistance, PlotLeftFingerPosition

#%%
q_sim = QuasistaticSimulator(is_quasi_dynamic=True)

#%% new "impedance robot" formulation, which I think is correct.
q = q0.copy()
q_sim.update_visualizer(q)
print(q0)
input("start?")
lambda_n_log = []
lambda_f_log = []
q_log = [q0.copy()]
qa_cmd_log = []

for i in range(n_steps):
    q_a_cmd = q_traj.value((i + 1) * h).squeeze()
    dq_a, dq_u, lambda_n, lambda_f, result = q_sim.step_lcp(q, q_a_cmd)

    # Update q.
    q += np.hstack([dq_u, dq_a])
    q_sim.update_visualizer(q)

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
t_sim1 = np.arange(n_steps + 1) * h
t_sim = np.arange(n_steps) * h
friction_log = np.zeros((n_steps, n_c))
contact_velocity_log = np.zeros((n_c, n_steps))
phi_log = np.array([calc_phi(q) for q in q_log])

for i in range(n_c):
    idx = i * 2
    friction_log[:, i] = lambda_f_log[:, idx] - lambda_f_log[:, idx + 1]

Jf = np.hstack((Jf_u, Jf_a))
Jn = np.hstack((Jn_u, Jn_a))
dq = (q_log[1:] - q_log[:-1])
v_tangent = (Jf.dot(dq.T / h)).T
v_normal = (dq / h).dot(Jn.T)


#%%
PlotForceDistance(t_sim, phi_log, lambda_n_log, friction_log,
                  figsize=(6, 4),
                  save_name="contact_force_distance_lcp.pdf")

PlotLeftFingerPosition(t_sim1, q_log, qa_cmd_log,
                       fig_size=(6, 3),
                       save_name="xy_cmd_vs_xy_true_lcp.pdf")
