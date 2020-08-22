
import numpy as np
import matplotlib.pyplot as plt

from quasistatic_sim import QuasistaticSimulator, CalcPhi, r, h, n_c


q_sim = QuasistaticSimulator()

#%% Anitescu
q0 = np.array([0, r, -r * 1.1, r * 1.1, 0])

q = q0.copy()
q_sim.UpdateVisualizer(q)
print(q0)
input("start?")
for i in range(100):
    dr = np.min([0.001 * i, 0.02])
    q_a_cmd = np.array([-r * 1.1 + dr, r * 1.1 - dr, 0.002 * i])
    dq_a, dq_u, result = q_sim.StepAnitescu(q, q_a_cmd)
    q += np.hstack([dq_u, dq_a])
    q_sim.UpdateVisualizer(q)
    print(i, dq_u, dq_a)
    input("contune?")