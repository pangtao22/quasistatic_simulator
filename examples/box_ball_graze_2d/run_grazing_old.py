import time

import numpy as np
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
from pydrake.all import PiecewisePolynomial

from qsim_old.simulator import QuasistaticSimulator
from qsim_old.problem_definition_graze import problem_definition

#%%
q_sim = QuasistaticSimulator(problem_definition, is_quasi_dynamic=True)


#%% plot the points
# viz.delete()
n_u = problem_definition['n_u']
h = problem_definition['h']

#%% define initial conditions and actuated trajectory.
q0 = np.array([0, 0, 0.1])

qa_knots = np.zeros((2, 2))
qa_knots[0] = q0[n_u:]
qa_knots[1] = [0.1, -0.1]

n_steps = 10
t_knots = [0, n_steps * h]
q_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)


#%% new "impedance robot" formulation, which I think is correct.
q = q0.copy()
lambda_n_log = []
lambda_f_log = []
q_log = [q0.copy()]
qa_cmd_log = []

for i in range(n_steps):
    q_a_cmd = q_traj.value((i + 1) * h).squeeze()
    dq_a, dq_u, lambda_n, lambda_f, result = q_sim.step_anitescu(q, q_a_cmd)

    # Update q.
    q += np.hstack([dq_u, dq_a])

    # logging.
    lambda_n_log.append(lambda_n)
    lambda_f_log.append(lambda_f)
    q_log.append(q.copy())
    qa_cmd_log.append(q_a_cmd)

    time.sleep(h)

q_log = np.array(q_log)
qa_cmd_log = np.array(qa_cmd_log)
lambda_n_log = np.array(lambda_n_log)
lambda_f_log = np.array(lambda_f_log)
