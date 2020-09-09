
from quasistatic_simulator import *
#%%
q_sim = QuasistaticSimulator(CreatePlantFor2dGripper, nd_per_contact=4)

#%%
q_a = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2])
q_u = np.array([1, 0, 0, 0, 0, -1, 0.])
q = np.hstack([q_u, q_a])
q_sim.UpdateConfiguration(q)
q_sim.DrawCurrentConfiguration()


#%%
q_a = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2])
q_u = np.array([2.0, 0.])
q = np.hstack([q_u, q_a])
q_sim.UpdateConfiguration(q)
q_sim.DrawCurrentConfiguration()