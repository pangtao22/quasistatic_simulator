import numpy as np
import matplotlib.pyplot as plt

from pydrake.common.eigen_geometry import Quaternion, AngleAxis

#%%
time_steps = [0.005, 0.01, 0.03, 0.025, 0.035]
file_names = ['q_10box_box0_h{}.npy'.format(a) for a in time_steps]
q_u0_logs = [np.load(file_name) for file_name in file_names]
q_u0_ground_truth = q_u0_logs[0]


def get_angle_from_quaternion(q: np.array):
    q /= np.linalg.norm(q)
    a = AngleAxis(Quaternion(q))
    return a.angle()


#%%
fig, axes = plt.subplots(4, 1, figsize=(5, 6), dpi=200)
for i, dt in enumerate(time_steps[:-1]):
    t = np.arange(len(q_u0_logs[i])) * time_steps[i]
    angles = [get_angle_from_quaternion(qu[:4].copy()) for qu in q_u0_logs[i]]
    for j, ax in enumerate(axes):
        if j == 3:
            ax.plot(t, angles, label="t={}".format(dt))
            break
        ax.plot(t, q_u0_logs[i][:, j + 4], label="t={}".format(dt))

plt.legend()
plt.show()




