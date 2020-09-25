import time

from pydrake.multibody.plant import CalcContactFrictionFromSurfaceProperties

from quasistatic_simulator import *
from meshcat_camera_utils import SetOrthographicCameraYZ

from sim_params_3link_arm import *

#%%
object_sdf_path = os.path.join("models", "box_yz_rotation_big.sdf")
# object_sdf_path = os.path.join("models", "sphere_yz_rotation_big.sdf")
# object_sdf_path = os.path.join("models", "sphere_yz_big.sdf")

q_sim = QuasistaticSimulator(CreatePlantFor2dArmWithMultipleObjects,
                             nd_per_contact=2,
                             object_sdf_path=[object_sdf_path],
                             joint_stiffness=Kq_a)
# SetOrthographicCameraYZ(q_sim.viz.vis)


#%%
q_a = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2])
q_u = np.array([1.7, 0.5, 0])
q_list = [q_u, q_a]
tau_u_ext = np.array([0., -10, 0])
q_sim.UpdateConfiguration(q_list)
q_sim.DrawCurrentConfiguration()

#%%
h = 0.01
n_steps = int(t_final / h)

input("start?")
for i in range(n_steps):
    q_a_cmd = q_a_traj.value(h * i).squeeze()
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
    time.sleep(h)
    input("next?")

#%%
(n_c, n_d, n_f, Jn_u_q, Jn_u_v, Jn_a, Jf_u_q, Jf_u_v, Jf_a, phi,
    contact_info_list) = q_sim.CalcContactJacobians(0.01)
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
    props_A = inspector.GetProximityProperties(sdp.id_A)
    props_B = inspector.GetProximityProperties(sdp.id_B)
    cf_A = props_A.GetProperty("material", "coulomb_friction")
    cf_B = props_B.GetProperty("material", "coulomb_friction")
    cf = CalcContactFrictionFromSurfaceProperties(cf_A, cf_B)
    print("coulomb friction: ", cf.static_friction(), cf.dynamic_friction())
    print("")



