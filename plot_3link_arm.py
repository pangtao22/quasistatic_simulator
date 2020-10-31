import numpy as np
import matplotlib.pyplot as plt


#%%
time_steps = [0.01, 0.025, 0.005]
# file_names = ['3link_box_mbp.npy', '3link_box_quasistatic_h0.025.npy']
file_names = ['3link_box_qa_mbp.npy', '3link_box_qa_quasistatic_h0.025.npy',
              '3link_box_qa_quasistatic_h0.005.npy']
q_a_logs = [np.load(file_name) for file_name in file_names]


#%%
fig, axes = plt.subplots(3, 1, figsize=(5, 6), dpi=200)
for i, dt in enumerate(time_steps):
    t = np.arange(len(q_a_logs[i])) * time_steps[i]
    for j, ax in enumerate(axes):
        ax.plot(t, q_a_logs[i][:, j], label="t={}".format(dt))

plt.legend()
plt.show()
