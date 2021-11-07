import time

import tqdm
import meshcat
import numpy as np
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
from pydrake.all import PiecewisePolynomial

from q_sim_old.simulator import QuasistaticSimulator
from q_sim_old.problem_definition_graze import problem_definition

#%%
q_sim = QuasistaticSimulator(problem_definition, is_quasi_dynamic=True)
viz = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")

#%% simulate a special case.
q = np.array([0, 0, 0.])
q_a_cmd = np.array([0.05, -0.02])
dq_a, dq_u, lambda_n, lambda_f, result = q_sim.step_lcp(q, q_a_cmd)
q += np.hstack([dq_u, dq_a])

#%% sample dynamics
# Sample actions between the box x \in [-0.05, 0.05] and y \in [-0.05, 0.05].
n = 1000
q_a_cmd = np.random.rand(n, 2) * 0.1 - 0.05
q_next = np.zeros((n, 3))

for i in tqdm.tqdm(range(n)):
    q0 = np.array([0, 0, 0.])
    dq_a, dq_u, lambda_n, lambda_f, result = q_sim.step_anitescu(q0, q_a_cmd[i])
    q_next[i] = q0 + np.hstack([dq_u, dq_a])



#%% plot the points
# viz.delete()
n_u = problem_definition['n_u']
h = problem_definition['h']
dynamics_lcp = np.hstack([q_a_cmd, q_next[:, :n_u]])  # [x_cmd, y_cmd, x_u_next]
discontinuity_lcp = np.hstack([q_a_cmd[:, 0][:, None],
                               q_next[:, 2][:, None],
                               q_next[:, 0][:, None]])
discontinuity2_lcp = np.hstack([q_a_cmd[:, 0][:, None],
                                q_a_cmd[:, 1][:, None],
                                q_next[:, 0][:, None]])


viz["dynamics_lcp"].set_object(
    meshcat.geometry.PointCloud(
        position=dynamics_lcp.T, color=np.ones_like(dynamics_lcp).T))
#
# viz["discontinuity_lcp"].set_object(
#     meshcat.geometry.PointCloud(
#         position=discontinuity_lcp.T, color=np.ones_like(dynamics_lcp).T))
#
# viz["discontinuity2_lcp"].set_object(
#     meshcat.geometry.PointCloud(
#         position=discontinuity2_lcp.T, color=np.zeros_like(dynamics_lcp).T))


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
    dq_a, dq_u, lambda_n, lambda_f, result = q_sim.step_lcp(q, q_a_cmd)

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




