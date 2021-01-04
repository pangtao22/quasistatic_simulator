import time

from examples.iiwa_block_stacking.simulation_parameters import *
from examples.iiwa_block_stacking.iiwa_block_stacking_mbp import run_sim


#%%
h_mbp = 0.001
create_initial_state_dictionary
# h_mbp = 0.001088
iiwa_log, schunk_log, object0_log = run_sim(q_iiwa_traj, x_schunk_traj,
                                            Kp_iiwa=Kp_iiwa,
                                            Kp_schunk=Kp_schunk,
                                            object_sdf_paths=object_sdf_paths,
                                            q_u0_list=q_u0_list,
                                            time_step=h_mbp)

#  save ground truth logs
na = 7
t_s = iiwa_log.sample_times()
q_iiwa_log = iiwa_log.data()[:na].T
q_schunk_log = schunk_log.data()[:2].T
q_a_log = np.hstack((q_iiwa_log, q_schunk_log))

# error_qa = np.zeros_like(q_a_log)
#
# for i, t in enumerate(t_s):
#     error_qa[i] = q_iiwa_traj.value(t).squeeze() - q_iiwa_log[i]

# np.save("qa_10cube_error_mbp_h{}".format(h_mbp), error_qa)
np.save("qa_10cube_q_mbp_h{}".format(h_mbp), q_a_log)
np.save("cube0_10cube_q_mbp_h{}".format(h_mbp), object0_log.data()[:7].T)
