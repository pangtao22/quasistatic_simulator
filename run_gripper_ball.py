import time

from quasistatic_simulator import *
from meshcat_camera_utils import SetOrthographicCameraYZ


#%%
Kq_a = np.ones(3) * 1000
q_sim = QuasistaticSimulator(CreatePlantFor2dGripper, nd_per_contact=2,
                             object_sdf_paths=None,
                             joint_stiffness=Kq_a)
# SetOrthographicCameraYZ(q_sim.viz.vis)

#%%
# TODO: note that the ordering of this q is different from the ordering in
#  the state vector of MutlbibodyPlant.
r = 0.1
q_a = np.array([0.1, -1.05*r, 1.05*r])
q_u = np.array([0, r])
q_list = [q_u, q_a]
q_sim.UpdateConfiguration(q_list)
q_sim.DrawCurrentConfiguration()

#%%
h = 0.01
tau_u_ext = np.array([0., -10])
n_steps = 50

# input("start?")
for i in range(n_steps):
    q_a_cmd = np.array([0.1 + np.max([-0.002 * i, -0.03]), -r * 0.9, r * 0.9])
    q_a_cmd_list = [None, q_a_cmd]
    tau_u_ext_list = [tau_u_ext, None]
    dq_u_list, dq_a_list = q_sim.StepAnitescu(
            q_list, q_a_cmd_list, tau_u_ext_list, h,
            is_planar=True,
            contact_detection_tolerance=0.01)

    # Update q
    q_sim.StepConfiguration(q_list, dq_u_list, dq_a_list, is_planar=False)
    q_sim.UpdateConfiguration(q_list)
    q_sim.DrawCurrentConfiguration()

    # logging
    # time.sleep(h * 10)
    # input("next?")

q_sim.PrintSimStatcs()

#%% Print contact information for one configuration.
n_c, n_d, n_f, Jn_v_list, Jf_v_list, phi, U, contact_info_list \
    = q_sim.CalcContactJacobians(0.01)

query_object = q_sim.scene_graph.get_query_output_port().Eval(
    q_sim.context_sg)
signed_distance_pairs = \
    query_object.ComputeSignedDistancePairwiseClosestPoints(0.01)

plant = q_sim.plant
inspector = query_object.inspector()
for i, sdp in enumerate(signed_distance_pairs):
    print("contact%d"%i)
    print(inspector.GetNameByGeometryId(sdp.id_A))
    print(inspector.GetNameByGeometryId(sdp.id_B))
    print(inspector.GetFrameId(sdp.id_A))
    print(inspector.GetFrameId(sdp.id_B))
    print("distance: ", sdp.distance)
    print("p_AC_a: ", sdp.p_ACa)
    print("p_BC_b: ", sdp.p_BCb)
    print("nhat_BA_W", sdp.nhat_BA_W)
    print("")
