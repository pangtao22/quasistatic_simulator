import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
from problem_definition_pinch import *

q_log = np.load("q_log.npy")
q_log_anitescu = np.load("q_log_anitescu.npy")
qa_cmd_log = np.load("q_cmd_log.npy")
t_sim1 = np.arange(n_steps + 1) * h

#%%
fig, axes = plt.subplots(2, 1, figsize=(6, 4), dpi=200)

labels = [r"$x_l$, LCP", r"$y_l$, LCP"]
labels_anitescu = [r"$x_l$, QP", r"$y_l$, QP"]
cmd_labels = [r"$\bar{x}_l$", r"$\bar{y}_l$"]
idx = [0, 2]
for i, ax in enumerate(axes):
    color = "red"
    color2 = np.array([0, 204, 163, 255]) / 255
    ax.step(t_sim1, q_log[:, 2 + idx[i]], where="post", color="blue",
            label=labels[i], linewidth=2)
    ax.step(t_sim1, q_log_anitescu[:, 2 + idx[i]], where="post",
            color=color, label=labels_anitescu[i], linewidth=1)
    ax.step(t_sim1[1:], qa_cmd_log[:, idx[i]], where="post", color=color2,
            label=cmd_labels[i], linewidth=1)

    ax.set_ylabel("[m]".format(i + 1))
    ax.grid(True)
    for t in t_contact_mode_change:
        ax.axvline(t, color='g', linestyle="--", linewidth=1.2)
    ax.legend(loc="upper right")

    if i < 1:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel("t [s]")

plt.tight_layout()
# plt.margins(0, 0)
plt.savefig("xy_cmd_vs_xy_true.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()