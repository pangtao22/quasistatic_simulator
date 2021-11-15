import numpy as np
import matplotlib.pyplot as plt
from typing import List
plt.rcParams.update({'font.size': 10})
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

from pydrake.math import RollPitchYaw
from pydrake.trajectories import PiecewisePolynomial
from pydrake.common.eigen_geometry import Quaternion, AngleAxis

#%%
# time_steps = [0.001, 0.01, 0.1, 0.2, 0.4]
time_steps = [1e-4, 0.001, 0.01, 0.1, 0.4]

n_quasistatic = len(time_steps)
# Load q_cube0
file_names = ['cube0_10cube_q_h{}.npy'.format(a) for a in time_steps]

# MBP
time_steps_mbp = [5e-4, 1e-3, 1e-4, 0.001085, 0.001086, 0.001087, 0.001088,
                  5e-5]
for time_step in time_steps_mbp:
    file_names.append("cube0_10cube_q_mbp_h{}.npy".format(time_step))
q_u0_logs = [np.load(file_name) for file_name in file_names]

# load q_iiwa
file_names = ['qa_10cube_q_h{}.npy'.format(a) for a in time_steps]
# MBP
for time_step in time_steps_mbp:
    file_names.append("qa_10cube_q_mbp_h{}.npy".format(time_step))
qa_logs = [np.load(file_name) for file_name in file_names]

time_steps_real = time_steps + time_steps_mbp
time_steps += [1e-2] * len(time_steps_mbp)


def get_angle_from_quaternion(q: np.array):
    q /= np.linalg.norm(q)
    a = AngleAxis(Quaternion(q))
    return a.angle()


def get_roll_from_quaternion(q: np.array):
    q /= np.linalg.norm(q)
    rpy = RollPitchYaw(Quaternion(q))
    return rpy.roll_angle()


t_list = []
for i, dt in enumerate(time_steps):
    t_list.append(np.arange(len(q_u0_logs[i])) * time_steps[i])


# compute integral error
def compute_error_integral(q_logs: List[np.array]):
    # converting trajectories to PiecewisePolynomials
    q_traj_list = []
    for i, dt in enumerate(time_steps):
        t = t_list[i]
        q_traj_list.append(
            PiecewisePolynomial.FirstOrderHold(t, q_logs[i].T))

    q_traj_gt = q_traj_list[-1]
    e_integral_list = []
    e_list = []
    for i, dt in enumerate(time_steps[:-1]):
        n_steps = len(q_logs[i])
        e_integral = np.zeros(n_steps)
        e = np.zeros(n_steps)
        for j in range(1, len(q_logs[i])):
            dq = q_traj_gt.value(j * dt).squeeze() - q_logs[i][j]
            e[j] = np.linalg.norm(dq)
            e_integral[j] = e_integral[j - 1] + dt * e[j]

        e_integral_list.append(e_integral)
        e_list.append(e)

    return e_integral_list, e_list


def get_label(i):
    if i >= n_quasistatic:
        label = r"MBP{}".format(time_steps_real[i])
    else:
        label = r"$h$={}s".format(dt)
    return label


#%% plot integral error
e_cube_integral_list, e_cube_list = compute_error_integral(q_u0_logs)
for i, dt in enumerate(time_steps[:-1]):
    label = get_label(i)
    plt.plot(t_list[i], e_cube_integral_list[i], label=label)
plt.ylabel(r"$\int \|q_u(t) - q_{u, GT}(t)\| dt$")
plt.xlabel(r"t [s]")
plt.legend()
plt.show()

#%%
e_a_integral_list, e_a_list = compute_error_integral(qa_logs)
for i, dt in enumerate(time_steps[:-1]):
    label = get_label(i)
    plt.plot(t_list[i], e_a_integral_list[i], label=label)

plt.ylabel(r"$\int \|q_a(t) - q_{a, GT}(t)\| dt$")
plt.xlabel(r"t [s]")
plt.legend()
plt.show()


#%% error as a function of time step
plt.figure(figsize=(6, 2.5))
e_cube_integral_final = [
    a[-1] for a, b in zip(e_cube_integral_list, e_a_integral_list)]
plt.grid(True)
plt.scatter(
    time_steps_real[:n_quasistatic],
    e_cube_integral_final[:n_quasistatic], label="Quasistatic")
plt.scatter(
    time_steps_real[n_quasistatic:-1],
    e_cube_integral_final[n_quasistatic:], label="MBP")

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlabel(r"$h$ [s]")
plt.ylabel(r"$\int \|q_u(t) - q_{u, GT}(t)\| dt$")
plt.tight_layout()
plt.savefig("error_vs_time_step.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()


#%%  Position of cube
fig, axes = plt.subplots(4, 1, figsize=(5, 6.2), dpi=200)

y_labels = [r'$x$ [m]', r'$y$ [m]', r'$z$ [m]', r'angle [rad]', r'roll [rad]']

for i, dt in enumerate(time_steps):
    if i >= n_quasistatic:
        if i == len(time_steps) - 1:
            # label = r"MBP{}s".format(time_steps_real[i])
            label = "GT"
        else:
            continue
    else:
        label = r"$h$={}s".format(dt)

    t = np.arange(len(q_u0_logs[i])) * time_steps[i]
    angles = [get_angle_from_quaternion(qu[:4].copy()) for qu in q_u0_logs[i]]
    roll_angles = [get_roll_from_quaternion(qu[:4].copy()) for qu in q_u0_logs[i]]
    for j, ax in enumerate(axes):
        if j == 3:
            ax.plot(t, angles, label=label)
        elif j == 4:
            ax.plot(t, roll_angles, label=label)
        else:
            ax.plot(t, q_u0_logs[i][:, j + 4], label=label)


for i, ax in enumerate(axes):
    ax.set_ylabel(y_labels[i])
    ax.set_axisbelow(True)
    ax.grid(True)
    if i < len(axes) - 1:
        plt.setp(ax.get_xticklabels(), visible=False)
axes[-1].set_xlabel("t [s]")
axes[2].legend(loc='lower right',  ncol=2)
plt.tight_layout()
plt.savefig("cube_pose.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()


