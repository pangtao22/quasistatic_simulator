import time

from quasistatic_simulation.quasistatic_simulator import *
from setup_environments import create_iiwa_plant_with_schunk
from examples.iiwa_block_stacking.trajectory_generation import *


#%%
q_sim = QuasistaticSimulator(
    create_iiwa_plant_with_schunk,
    nd_per_contact=4,
    object_sdf_paths=object_sdf_paths,
    joint_stiffness=Kq_a,
    internal_vis=True)

(model_instance_indices_u,
 model_instance_indices_a) = q_sim.get_model_instance_indices()

t_start = q_iiwa_traj.start_time()
q0_dict = create_initial_state_dictionary(
    q0_iiwa=q_iiwa_traj.value(t_start).squeeze(),
    q0_schunk=q_schunk_traj.value(t_start).squeeze(),
    q_u0_list=q_u0_list,
    model_instance_indices_u=model_instance_indices_u,
    model_instance_indices_a=model_instance_indices_a)

idx_iiwa, idx_schunk = model_instance_indices_a

#%%
q_sim.viz.vis["drake"]["contact_forces"].delete()
q_sim.update_configuration(q0_dict)
q_sim.draw_current_configuration()

#%%
h = 0.2
q_iiwa_traj.shiftRight(-h)
q_schunk_traj.shiftRight(-h)

q_log = [q0_dict]
q_a_cmd_log = []

input("start?")
n_steps = round(q_iiwa_traj.end_time() / h)

for i in range(n_steps):
    q_a_cmd_dict = {idx_iiwa: q_iiwa_traj.value(h * i).squeeze(),
                    idx_schunk: q_schunk_traj.value(h * i).squeeze()}
    tau_ext_dict = q_sim.calc_gravity_for_unactuated_models()
    q_dict = q_sim.step_anitescu(
            q_a_cmd_dict, tau_ext_dict, h,
            is_planar=False,
            contact_detection_tolerance=0.005)
    q_sim.draw_current_configuration()

    q_a_cmd_log.append(q_a_cmd_dict)
    q_log.append(q_dict)

    # time.sleep(h)
    # print("t = ", i * h)
    # input("step?")


#%%
def extract_log_for_object(
        q_log: List[Dict[ModelInstanceIndex, np.array]],
        model: ModelInstanceIndex):
    n = len(q_log)
    m = len(q_log[0][model])
    q_i_log = np.zeros((n, m))
    for t, q_t in enumerate(q_log):
        q_i_log[t] = q_t[model]
    return q_i_log

q_iiwa_log = extract_log_for_object(q_log, idx_iiwa)
q_schunk_log = extract_log_for_object(q_log, idx_schunk)
q_iiwa_cmd_log = extract_log_for_object(q_a_cmd_log, idx_iiwa)
q_schunk_cmd_log = extract_log_for_object(q_a_cmd_log, idx_schunk)
q_u0_log = extract_log_for_object(q_log, q_sim.models_unactuated[0])


#%%
#
# for i in range(len(q_log[0])):
#     print(i, q_log[0][i][-3:], q_log[-1][i][-3:])

error_qa = np.zeros_like(q_a_log[:, :7])
t_s = np.arange(n_steps) * h
for i, t in enumerate(t_s):
    error_qa[i] = q_iiwa_traj.value(t).squeeze() - q_a_log[i, :7]

np.save("qa_10cube_error_h{}".format(h), error_qa)
np.save("qa_10cube_q_h{}".format(h), q_a_log)  # IIWA only
np.save("cube0_10cube_q_h{}".format(h), q_u0_log)

#%% log playback
# stride = 500
# for i in range(0, len(q_log), stride):
#     q_sim.update_configuration(q_log[i])
#     q_sim.draw_current_configuration()
#     time.sleep(h * stride)


#%% with System wrapper.
