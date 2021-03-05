from collections import namedtuple
from typing import List, Union, Dict

from pydrake.all import (SignedDistancePair_, AutoDiffXd, ModelInstanceIndex,
                         Sphere, GeometryId)
from pydrake.autodiffutils import (initializeAutoDiff, autoDiffToValueMatrix,
                                   autoDiffToGradientMatrix)

from quasistatic_simulation.environment_setup import (RobotInfo,
    create_plant_with_robots_and_objects)
from quasistatic_simulation.quasistatic_simulator import QuasistaticSimulator

from examples.planar_hand_ball.run_planar_hand import *


#%%
SignedDistancePairTuple = namedtuple("SignedDistancePairTuple",
                                     ["id_A", "id_B", "p_ACa", "p_BCb",
                                      "distance", "nhat_BA_W"])


class TrajectoryOptimizer:
    def __init__(self):
        #TODO: Most of these are imported from run_planar_hand.py. They
        # should come from... elsewhere...

        # MultibodyPlant.
        q_sim = QuasistaticSimulator(
            robot_info_dict=robot_info_dict,
            object_sdf_paths=object_sdf_dict,
            gravity=np.array([0, 0, -10]),
            nd_per_contact=2,
            sim_settings=sim_settings,
            internal_vis=False)
        self.q_sim = q_sim
        self.plant = q_sim.plant
        self.sg = q_sim.scene_graph
        self.context_plant = q_sim.context_plant
        self.context_sg = q_sim.context_sg

        self.plant_ad = self.plant.ToAutoDiffXd()
        self.context_plant_ad = self.plant_ad.CreateDefaultContext()

        inspector = self.sg.model_inspector()

        # Extract collision geometry for the object sphere.
        model_sphere = q_sim.models_unactuated[0]
        body_object = self.plant.GetBodyByName("sphere", model_sphere)
        collision_geometries = self.plant.GetCollisionGeometriesForBody(
            body_object)
        assert len(collision_geometries) == 1
        object_collision_g_id = collision_geometries[0]

        ground_collsion_g_id = self.plant.GetCollisionGeometriesForBody(
            self.plant.GetBodyByName("ground"))[0]

        # Get collision pairs that include the candidate. The first geometry
        # in each pair is always the collision geometry of the object being
        # manipulated.
        self.collision_pairs = []
        for cc in inspector.GetCollisionCandidates():
            if cc[0] == object_collision_g_id:
                if cc[1] != ground_collsion_g_id:
                    self.collision_pairs.append(
                        (object_collision_g_id, cc[1]))
            elif cc[1] == object_collision_g_id:
                if cc[0] != ground_collsion_g_id:
                    self.collision_pairs.append(
                        (object_collision_g_id, cc[0]))

        # dictionary of geometry to frame transforms.
        self.X_FG_dict = {object_collision_g_id:
                          inspector.GetPoseInFrame(object_collision_g_id)}
        for _, g_id in self.collision_pairs:
            self.X_FG_dict[g_id] = inspector.GetPoseInFrame(g_id)

        self.g_id_to_mbp_body_id_map = {
            g_id: self.plant.GetBodyFromFrameId(
                inspector.GetFrameId(g_id)).index()
            for g_id in inspector.GetAllGeometryIds()}

    def update_configuration(
            self, q_ad_dict: Dict[ModelInstanceIndex, np.ndarray]):
        assert len(q_ad_dict) == len(self.q_sim.models_all)

        # Update configuration in self.q_sim.context_plant.
        q_dict = {model: autoDiffToValueMatrix(q_ad).squeeze()
                  for model, q_ad in q_ad_dict.items()}
        self.q_sim.update_configuration(q_dict)

        # Update state in self.plant_context_ad.
        for model_instance_idx, q_ad in q_ad_dict.items():
            self.plant_ad.SetPositions(
                self.context_plant_ad, model_instance_idx, q_ad)

    def detect_collision(self):
        """
        Run collision detection using SceneGraph.
        :return:
        """
        query_object = self.sg.get_query_output_port().Eval(self.context_sg)
        signed_distance_pairs = [
            query_object.ComputeSignedDistancePairClosestPoints(*geometry_pair)
            for geometry_pair in self.collision_pairs]
        return signed_distance_pairs

    def get_geometry_pose_in_world_frame(self, g_id: GeometryId):
        body = self.plant_ad.get_body(self.g_id_to_mbp_body_id_map[g_id])
        X_FG = self.X_FG_dict[g_id]
        X_WF = self.plant_ad.EvalBodyPoseInWorld(self.context_plant_ad, body)
        return X_WF.multiply(X_FG.cast[AutoDiffXd]())

    def detect_collision_ad(self):
        """
        Run collision manually for all contact pairs in self.collision_pairs.
        All collision geometries need to be spheres.
        :return:
        """
        inspector = self.sg.model_inspector()
        signed_distance_pairs = []

        for g_id_A, g_id_B in self.collision_pairs:
            # Make sure that geometries are spheres.
            shape_A = inspector.GetShape(g_id_A)
            shape_B = inspector.GetShape(g_id_B)
            assert isinstance(shape_A, Sphere) and isinstance(shape_B, Sphere)
            r_A = shape_A.radius()
            r_B = shape_B.radius()
            X_WA = self.get_geometry_pose_in_world_frame(g_id_A)
            X_WB = self.get_geometry_pose_in_world_frame(g_id_B)

            p_AoW_W = X_WA.translation()
            p_BoW_W = X_WB.translation()

            d = p_AoW_W - p_BoW_W
            d_norm = np.sqrt((d ** 2).sum())
            distance = d_norm - r_A - r_B
            nhat_BA_W = d / d_norm
            p_AcA_W = -nhat_BA_W * r_A
            p_BcB_W = nhat_BA_W * r_B
            p_AcA_A = X_WA.rotation().inverse().multiply(p_AcA_W)
            p_BcB_B = X_WB.rotation().inverse().multiply(p_BcB_W)

            sdp = SignedDistancePairTuple(
                id_A=g_id_A,
                id_B=g_id_B,
                p_ACa=p_AcA_A,
                p_BCb=p_BcB_B,
                distance=distance,
                nhat_BA_W=nhat_BA_W)

            signed_distance_pairs.append(sdp)

        return signed_distance_pairs

    def print_geometry_info(self):
        inspector = self.sg.model_inspector()
        # Collision pairs involving the object.
        for g1, g2 in self.collision_pairs:
            print(g1, g2, inspector.GetName(g1), inspector.GetName(g2))

        print()
        for f_id in inspector.all_frame_ids():
            print(f_id, inspector.GetName(f_id),
                  inspector.NumGeometriesForFrame(f_id))


traj_opt = TrajectoryOptimizer()

#%%
name_to_model_map = traj_opt.q_sim.get_robot_name_to_model_instance_dict()
q0_dict = {name_to_model_map[name]: q0 for name, q0 in q0_dict_str.items()}
q0_ad_dict = {model: initializeAutoDiff(q) for model, q in q0_dict.items()}

traj_opt.update_configuration(q0_ad_dict)

signed_distance_pairs = traj_opt.detect_collision()
signed_distance_pairs_ad = traj_opt.detect_collision_ad()

#%% test hand-written collision detection.
for sdp, sdp_ad in zip(signed_distance_pairs, signed_distance_pairs_ad):
    if sdp.id_A == sdp_ad.id_A:
        assert sdp.id_B == sdp_ad.id_B
        assert np.allclose(sdp.p_ACa,
                           autoDiffToValueMatrix(sdp_ad.p_ACa).squeeze())
        assert np.allclose(sdp.p_BCb,
                           autoDiffToValueMatrix(sdp_ad.p_BCb).squeeze())
        assert np.allclose(sdp.distance, sdp_ad.distance.value())
        assert np.allclose(sdp.nhat_BA_W,
                           autoDiffToValueMatrix(sdp_ad.nhat_BA_W).squeeze())
    elif sdp.id_A == sdp_ad.id_B:
        assert sdp.id_B == sdp_ad.id_A
        assert np.allclose(sdp.p_BCb,
                           autoDiffToValueMatrix(sdp_ad.p_ACa).squeeze())
        assert np.allclose(sdp.p_ACa,
                           autoDiffToValueMatrix(sdp_ad.p_BCb).squeeze())
        assert np.allclose(sdp.distance, sdp_ad.distance.value())
        assert np.allclose(-sdp.nhat_BA_W,
                           autoDiffToValueMatrix(sdp_ad.nhat_BA_W).squeeze())
    else:
        raise RuntimeError("hand-written collision detection is wrong.")

