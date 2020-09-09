from quasistatic_simulator import *

q_sim = QuasistaticSimulator(CreatePlantFor2dGripper, nd_per_contact=2)

#%%
q_a = np.array([0.10, -0.11, 0.11])
q_u = np.array([0, 0.1])
q = np.hstack([q_u, q_a])
q_sim.UpdateConfiguration(q)
q_sim.DrawCurrentConfiguration()

#%%
n_c, n_d, n_f, Jn_u, Jn_a, Jf_u, Jf_a = q_sim.CalcContactJacobians(0.01)
query_object = q_sim.scene_graph.get_query_output_port().Eval(q_sim.context_sg)
signed_distance_pairs = \
    query_object.ComputeSignedDistancePairwiseClosestPoints(0.01)


#%%
plant = q_sim.plant
robot_model = q_sim.models_actuated[0]
object_model = q_sim.models_unactuated[0]


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

