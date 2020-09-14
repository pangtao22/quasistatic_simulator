import time

from quasistatic_simulator import *

#%%
q_sim = QuasistaticSimulator(CreatePlantFor2dArmWithFree3DBox, nd_per_contact=8)


#%%
q_a = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2])
q_u = np.array([1, 0, 0, 0, 0, 1.7, 0.5])
q = np.hstack([q_u, q_a])
q_sim.UpdateConfiguration(q)
q_sim.DrawCurrentConfiguration()


#%%
h = 0.001
tau_u_ext = np.array([0, 0, 0, 0., 0, -5])
n_steps = int(0.5 / h)

input("start?")
for i in range(n_steps):
    q_a_cmd = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2]) + h * i
    dq_a, dq_u, beta, constraint_values, result = q_sim.StepAnitescu3D(
        q, q_a_cmd, tau_u_ext, h)

    # Update q
    q += np.hstack([dq_u, dq_a])
    q[:4] / np.linalg.norm(q[:4])  # normalize quaternion
    q_sim.UpdateConfiguration(q)
    q_sim.DrawCurrentConfiguration()
    print("qu: ", q[:7])
    print("dq_u", dq_u)
    # logging
    # time.sleep(h)
    input("next?")


#%%
n_c, n_d, n_f, Jn_u_q, Jn_u_v, Jn_a, Jf_u_q, Jf_u_v, Jf_a, phi = \
    q_sim.CalcContactJacobians(0.01)
query_object = q_sim.scene_graph.get_query_output_port().Eval(q_sim.context_sg)
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


#%%
body = plant.GetBodyByName("base_link")
J_WBi = plant.CalcJacobianTranslationalVelocity(
    context=q_sim.context_plant,
    with_respect_to=JacobianWrtVariable.kV,
    frame_B=body.body_frame(),
    p_BoBi_B=np.zeros(3),
    frame_A=plant.world_frame(),
    frame_E=plant.world_frame())


#%%
sdp = signed_distance_pairs[5]
inspector.GetPoseInFrame(sdp.id_B).matrix()





