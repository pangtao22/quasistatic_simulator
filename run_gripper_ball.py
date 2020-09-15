import time

from quasistatic_simulator import *
from meshcat_camera_utils import SetOrthographicCameraYZ


#%%
Kq_a = np.ones(3) * 1000
q_sim = QuasistaticSimulator(CreatePlantFor2dGripper, nd_per_contact=2,
                             object_sdf_path=None,
                             joint_stiffness=Kq_a)
# SetOrthographicCameraYZ(q_sim.viz.vis)

#%%
# TODO: note that the ordering of this q is different from the ordering in
#  the state vector of MutlbibodyPlant.
r = 0.1
q_a = np.array([0.1, -1.05*r, 1.05*r])
q_u = np.array([0, r])
q = np.hstack([q_u, q_a])
q_sim.UpdateConfiguration(q)
q_sim.DrawCurrentConfiguration()

#%%
h = 0.01
tau_u_ext = np.array([0., -10])
n_steps = 50

# input("start?")
for i in range(n_steps):
    # PrintAllContacts(q_sim)
    # dr = np.min([0.001 * i, 0.02])
    # q_a_cmd = np.array([-r * 1.1 + dr, r * 1.1 - dr, -0.002 * i])
    q_a_cmd = np.array([0.1 + np.max([-0.002 * i, -0.03]), -r * 0.9, r * 0.9])
    dq_a, dq_u, beta, constraint_values, result, contact_results = \
        q_sim.StepAnitescu(
            q, q_a_cmd, tau_u_ext, h,
            is_planar=True,
            contact_detection_tolerance=0.01)

    # Update q
    q += np.hstack([dq_u, dq_a])
    q_sim.UpdateConfiguration(q)
    q_sim.DrawCurrentConfiguration()

    # logging
    # time.sleep(h * 10)
    input("next?")


#%% Print contact information for one configuration.
n_c, n_d, n_f, Jn_u_q, Jn_u_v, Jn_a, Jf_u_q, Jf_u_v, Jf_a, phi = \
    q_sim.CalcContactJacobians(0.01)

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
