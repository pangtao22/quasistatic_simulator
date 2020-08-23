import numpy as np

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

