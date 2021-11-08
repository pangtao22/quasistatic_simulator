import time

import numpy as np
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
from pydrake.all import PiecewisePolynomial

from qsim_old.simulator import QuasistaticSimulator
from qsim_old.problem_definition_pinch import problem_definition
from plotting import PlotForceDistance, PlotLeftFingerPosition

#%%
q_sim = QuasistaticSimulator(problem_definition, is_quasi_dynamic=True,
                             visualize=True)

#%% new "impedance robot" formulation, which I think is correct.
# define actuated trajectory
r = problem_definition['r']
h = problem_definition['h']

q0 = np.array([0, r, -1.06*r, 1.06*r, 0])

qa_knots = np.zeros((4, 3))
qa_knots[0] = q0[2:]
qa_knots[1] = [-0.9*r, 0.9*r, 0]
qa_knots[2] = [-0.9*r, 0.9*r, -0.03]
qa_knots[3] = [-0.9*r, 0.9*r, -0.03]

n_steps = 35
t_knots = [0, 8*h, (8 + 15)*h, n_steps * h]
q_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)

t_contact_mode_change = [0.03, 0.13, 0.23]

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
n_c = problem_definition['n_c']
calc_phi = problem_definition['calc_phi']
Jf_u = problem_definition['Jf_u']
Jf_a = problem_definition['Jf_a']
Jn_u = problem_definition['Jn_u']
Jn_a = problem_definition['Jn_a']

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
                  t_contact_mode_change,
                  figsize=(6, 4),
                  save_name="contact_force_distance_lcp.pdf")

PlotLeftFingerPosition(t_sim1, q_log, qa_cmd_log, t_contact_mode_change,
                       fig_size=(6, 3),
                       save_name="xy_cmd_vs_xy_true_lcp.pdf")
