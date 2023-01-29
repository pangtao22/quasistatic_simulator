import pickle

import numpy as np
import matplotlib.pyplot as plt

from examples.log_comparison import calc_error_integral

from pydrake.all import PiecewisePolynomial

plt.rcParams.update({"font.size": 10})
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)


#%% Reference trajectory (copied from run_iiwa_traj_following.py)
nq_a = 7
qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [0, 0, 0, -1.70, 0, 1.0, 0]
qa_knots[1] = qa_knots[0] + 0.5
q_iiwa_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=[0, 10],
    samples=qa_knots.T,
    sample_dot_at_start=np.zeros(nq_a),
    sample_dot_at_end=np.zeros(nq_a),
)


#%% Load data
time_steps = [0.0001, 0.001, 0.01, 0.1, 0.5]

q_iiwa_dict_list = []
for file_name in [f"q_iiwa_h_{a}.pkl" for a in time_steps]:
    with open(f"./data/{file_name}", "rb") as f:
        q_iiwa_dict_list.append(pickle.load(f))


#%% plot error comparison
n_trials = len(time_steps)
e_mbp = np.zeros(n_trials)
e_quasi_static = np.zeros(n_trials)
for i, q_iiwa_dict in enumerate(q_iiwa_dict_list):
    e, _, _ = calc_error_integral(
        q_knots=q_iiwa_dict["q_iiwa_mbp"],
        t=q_iiwa_dict["t_mbp"],
        q_gt_traj=q_iiwa_traj,
    )
    e_mbp[i] = e

    e, _, _ = calc_error_integral(
        q_knots=q_iiwa_dict["q_iiwa_quasi_static"],
        t=q_iiwa_dict["t_quasi_static"],
        q_gt_traj=q_iiwa_traj,
    )
    e_quasi_static[i] = e

T = q_iiwa_traj.end_time() - q_iiwa_traj.start_time()
e_mbp /= T
e_quasi_static /= T

#%%
plt.figure(figsize=(6, 3), dpi=300)
plt.scatter(time_steps, e_mbp, label="MBP")
plt.scatter(time_steps, e_quasi_static, label="Quasi-static")

label = (
    r"$\Delta(q^\mathrm{u}_\mathrm{QS/MBP}, q^\mathrm{u}_\mathrm{GT})$ ["
    r"rad]"
)

plt.xscale("log")
plt.yscale("log")
plt.ylabel(label)
plt.xlabel(r"$h$ [s]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("error_vs_time_step.pdf", bbox_inches="tight", pad_inches=0.01)
plt.show()
