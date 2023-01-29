import pickle
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from examples.log_comparison import calc_pose_error_integral

plt.rcParams.update({"font.size": 10})
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from pydrake.common.eigen_geometry import Quaternion, AngleAxis

#%% Load data
# time_steps = [0.001, 0.01, 0.1, 0.2, 0.4]
time_steps = [5e-5, 0.0001, 0.001, 0.01, 0.1, 0.5]


q_u_box0_dict_list = []
for file_name in [f"q_u_box0_h_{a}.pkl" for a in time_steps]:
    with open(f"./data/{file_name}", "rb") as f:
        q_u_box0_dict_list.append(pickle.load(f))


t_gt = q_u_box0_dict_list[0]["t_mbp"]
q_u_box0_gt = q_u_box0_dict_list[0]["q_u_box0_mbp"][:7].T


#%% plot integral error
n_trials = len(time_steps)
e_angle_mbp_list = np.zeros(n_trials)
e_angle_quasi_static_list = np.zeros(n_trials)
e_xyz_mbp_list = np.zeros(n_trials)
e_xyz_quasi_static_list = np.zeros(n_trials)
for i, q_u_box0_dict in enumerate(q_u_box0_dict_list):
    # Quasi-static
    (e_angle, _, _, e_xyz, _, _,) = calc_pose_error_integral(
        pose_list_1=q_u_box0_dict["q_u_box0_quasi_static"][:7].T,
        t1=q_u_box0_dict["t_quasi_static"],
        pose_list_2=q_u_box0_gt,
        t2=t_gt,
    )
    e_angle_quasi_static_list[i] = e_angle
    e_xyz_quasi_static_list[i] = e_xyz

    # MBP
    (e_angle, _, _, e_xyz, _, _,) = calc_pose_error_integral(
        pose_list_1=q_u_box0_dict["q_u_box0_mbp"][:7].T,
        t1=q_u_box0_dict["t_mbp"],
        pose_list_2=q_u_box0_gt,
        t2=t_gt,
    )
    e_angle_mbp_list[i] = e_angle
    e_xyz_mbp_list[i] = e_xyz


T = t_gt[-1]
e_angle_mbp_list /= T
e_angle_quasi_static_list /= T
e_xyz_mbp_list /= T
e_xyz_quasi_static_list /= T

#%%
_, axes = plt.subplots(1, 2, figsize=(12, 3), dpi=600)
axes[0].scatter(time_steps[1:], e_angle_mbp_list[1:], label="MBP")
axes[0].scatter(time_steps, e_angle_quasi_static_list, label="Quasi-static")
axes[1].scatter(time_steps[1:], e_xyz_mbp_list[1:], label="MBP")
axes[1].scatter(time_steps, e_xyz_quasi_static_list, label="Quasi-static")

a = r"$\Delta(q^\mathrm{u}_\mathrm{QS/MBP}, q^\mathrm{u}_\mathrm{GT})$"
axes_labels = [a + ", angular [rad]", a + ", translational [m]"]
for ax, ax_label in zip(axes, axes_labels):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(ax_label)
    ax.set_xlabel(r"$h$ [s]")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("error_vs_time_step.pdf", bbox_inches="tight", pad_inches=0.01)
plt.show()


#%% Pose of box0
fig, axes = plt.subplots(4, 1, figsize=(5, 6.2), dpi=300)

y_labels = [r"$x$ [m]", r"$y$ [m]", r"$z$ [m]", r"angle [rad]", r"roll [rad]"]


def get_angle_from_quaternion(q: np.array):
    q /= np.linalg.norm(q)
    a = AngleAxis(Quaternion(q))
    return a.angle()


def plot_q_u_trj(t: np.ndarray, q_u0_log: np.ndarray, label: str):
    angles = [get_angle_from_quaternion(qu[:4].copy()) for qu in q_u0_log]
    for j, ax in enumerate(axes):
        if j == 3:
            ax.plot(t, angles, label=label)
        else:
            ax.plot(t, q_u0_log[:, j + 4], label=label)


# Plot quasi-static trajectories
for i, dt in enumerate(time_steps):
    q_u_box0_dict = q_u_box0_dict_list[i]
    plot_q_u_trj(
        q_u0_log=q_u_box0_dict["q_u_box0_quasi_static"][:7].T,
        t=q_u_box0_dict["t_quasi_static"],
        label=r"$h$={}s".format(dt),
    )

# Plot ground truth
plot_q_u_trj(q_u0_log=q_u_box0_gt, t=t_gt, label="GT")


for i, ax in enumerate(axes):
    ax.set_ylabel(y_labels[i])
    ax.set_axisbelow(True)
    ax.grid(True)
    if i < len(axes) - 1:
        plt.setp(ax.get_xticklabels(), visible=False)
axes[-1].set_xlabel("t [s]")
axes[-1].legend(loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig("box0_pose.pdf", bbox_inches="tight", pad_inches=0.01)
plt.show()
