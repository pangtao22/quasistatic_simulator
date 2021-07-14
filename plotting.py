import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

from problem_definition_pinch import *



#%%
def PlotForceDistance(t_sim, phi_log, lambda_n_log, friction_log,
                      figsize, save_name):
    """
    plot normal, friction force and distance
    :return:
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=200)
    for i, ax in zip([0, 2], axes):
        ax2 = ax.twinx()
        color = "blue"
        ax2.step(t_sim, phi_log[:-1, i], 'o', markersize=3, where="post",
                 color=color)
        ax2.set_ylabel(r"$\phi_{}$ [m]".format(i + 1), color=color)
        ax2.set_ylim([-0.001, 0.0065])
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_axisbelow(True)
        ax2.grid(True)

        color = "red"
        color2 = np.array([0, 204, 163, 255]) / 255
        ax.step(t_sim, lambda_n_log[:, i], where="post", color=color,
                label=r"$c_{n_%d}$" % (i+1), linewidth=1)

        if i != 2:
            friction = (friction_log[:, 0] + friction_log[:, 1]) / 2
        else:
            friction = friction_log[:, i]
        ax.step(t_sim, friction, where="post", color=color2, linewidth=1,
                label=r"$c_{f_%d}$" % (i+1))
        ax.set_ylabel("contact force [N]".format(i + 1), color=color)
        for t in t_contact_mode_change:
            ax.axvline(t, color='g', linestyle="--", linewidth=1.2)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.legend(loc="center right")

        if i < n_c - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    axes[-1].set_xlabel("t [s]")

    plt.tight_layout()
    # plt.margins(0, 0)
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.close()


def PlotVelocity(t_sim, v_tangent, v_normal):
    """
    plot normal and tangent velocity
    :return:
    """
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), dpi=200)
    for i, ax in zip([0, 2], axes):
        ax.step(t_sim, v_tangent[:, i * 2], where="post", label="t")
        ax.step(t_sim, v_normal[:, i], where="post", label="n")
        ax.set_ylabel(r"$\tilde{v}_%d$ [m/s]" % (i + 1))
        ax.grid(True)
        ax.legend()
        for t in t_contact_mode_change:
            ax.axvline(t, color='g', linestyle="--", linewidth=1.2)

        if i < n_c - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        if i == n_c - 1:
            ax.set_ylim([-0.1, 0.1])

    axes[-1].set_xlabel("t [s]")
    plt.show()
    plt.close()


def PlotLeftFingerPosition(t_sim1, q_log, qa_cmd_log,
                           fig_size, save_name):
    """
    plot xy cmd vs xy true
    :return:
    """
    fig, axes = plt.subplots(2, 1, figsize=fig_size, dpi=200)

    labels = [r"$x_l$", r"$y_l$"]
    cmd_labels = [r"$\bar{x}_l$", r"$\bar{y}_l$"]
    idx = [0, 2]
    for i, ax in enumerate(axes):
        color = "red"
        color2 = np.array([0, 204, 163, 255]) / 255
        ax.step(t_sim1, q_log[:, 2 + idx[i]], where="post", color=color,
                label=labels[i], linewidth=1)
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
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.01)
    plt.show()
