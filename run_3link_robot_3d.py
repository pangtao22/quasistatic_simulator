import time
import copy

from quasistatic_simulator import *
from sim_params_3link_arm import *
from meshcat_camera_utils import SetOrthographicCameraYZ


#%%
q_sim = QuasistaticSimulator(
    Create2dArmPlantWithMultipleObjects,
    nd_per_contact=4,
    object_sdf_paths=[box3d_big_sdf_path],
    joint_stiffness=Kq_a)
# SetOrthographicCameraYZ(q_sim.viz.vis)

#%%
q_u1_0 = np.array([1, 0, 0, 0, 0, 1.7, 0.5])
# q_u2_0 = np.array([1, 0, 0, 0, 0.4, 2.5, 0.25])
q0_list = [q_u1_0, q_a0]

q_sim.update_configuration(q0_list)
q_sim.draw_current_configuration()

#%%
h = 0.005
tau_u_ext = np.array([0, 0, 0, 0., 0, -50])
n_steps = int(t_final / h)
q_list = copy.deepcopy(q0_list)
qa_log = []

input("start?")
for i in range(n_steps):
    q_a_cmd = q_a_traj.value(h * i).squeeze()
    q_a_cmd_list = [None, q_a_cmd]
    tau_u_ext_list = [tau_u_ext, None]
    dq_u_list, dq_a_list = q_sim.step_anitescu(
            q_list, q_a_cmd_list, tau_u_ext_list, h,
            is_planar=False,
            contact_detection_tolerance=0.01)

    # Update q
    q_sim.step_configuration(q_list, dq_u_list, dq_a_list, is_planar=False)
    q_sim.update_configuration(q_list)
    q_sim.draw_current_configuration()
    # print("qu: ", q[:7])
    # print("dq_u", dq_u)
    # logging
    # input("next?")
    qa_log.append(q_list[-1].copy())


#%%
t_s = np.arange(n_steps)
qa_log = np.array(qa_log)
error_qa = np.zeros_like(qa_log)

for i, t in enumerate(t_s):
    error_qa[i] = q_a_traj.value(t).squeeze() - qa_log[i]

np.save("3link_box_error_quasistatic_h{}".format(h), error_qa)
np.save("3link_box_qa_quasistatic_h{}".format(h), qa_log)


