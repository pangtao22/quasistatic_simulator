import tqdm
import meshcat
import numpy as np

from qsim_old.simulator import QuasistaticSimulator
from qsim_old.problem_definition_graze import problem_definition

#%% sim old
q_sim = QuasistaticSimulator(problem_definition, is_quasi_dynamic=True)
viz = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")


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


