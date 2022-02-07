from collections import namedtuple
from typing import List, Union, Dict

from pydrake.all import (AutoDiffXd, ModelInstanceIndex,
                         Sphere, GeometryId, JacobianWrtVariable)
from pydrake.autodiffutils import (initializeAutoDiff, autoDiffToValueMatrix,
                                   autoDiffToGradientMatrix,
                                   initializeAutoDiffGivenGradientMatrix)

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.snopt import SnoptSolver

from qsim.simulator import QuasistaticSimulator

from examples.planar_hand_ball.run_planar_hand import *

solver_snopt = SnoptSolver()


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
        self.n_q = self.plant.num_positions()

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

        self.name_to_model_idx_map = q_sim.get_model_instance_name_to_index_map()

    def update_configuration(
            self, q_ad_dict: Dict[ModelInstanceIndex, np.ndarray]):
        assert len(q_ad_dict) == len(self.q_sim.models_all)

        # Update configuration in self.q_sim.context_plant.
        q_dict = {model: autoDiffToValueMatrix(q_ad).squeeze()
                  for model, q_ad in q_ad_dict.items()}
        self.q_sim.update_mbp_positions(q_dict)

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

    @staticmethod
    def calc_tangent_vectors_yz(normal: np.ndarray):
        """

        :param normal: (3,) unit vector, coordinates of a normal vector in yz
            plane.
        :return:
        """
        tangents = np.zeros((2, normal.size), dtype=normal.dtype)
        tangents[0] = [0, -normal[2], normal[1]]
        tangents[1] = -tangents[0]
        return tangents

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

    def q_dict_to_vec(self,
                      q_dict: Dict[Union[str, ModelInstanceIndex], np.ndarray]):
        for q_sample in q_dict.values():
            break
        q_vec = np.zeros(self.q_sim.n_v, dtype=q_sample.dtype)

        for model, q_i in q_dict.items():
            if isinstance(model, str):
                model = self.name_to_model_idx_map[model]
            indices = self.q_sim.velocity_indices[model]
            q_vec[indices] = q_i

        return q_vec

    def vec_to_q_dict(self, q_vec: np.ndarray):
        q_dict = dict()
        for model in self.q_sim.models_all:
            indices = self.q_sim.velocity_indices[model]
            q_dict[model] = q_vec[indices]
        return q_dict

    def eval_dynamics_constraint(self, input: np.ndarray):
        """
        Constraint evaluation function for MathematicalProgram.
        :param input: [q, v, v_next, tau_a].
        :return:
        """
        n_q = self.plant.num_positions()
        n_v = self.plant.num_velocities()
        n_a = self.plant.num_actuated_dofs()

        assert input.dtype != float
        assert input.size == n_q + 2 * n_v + n_a
        assert n_q == n_v
        q = input[:n_q]
        v = input[n_q: n_q + n_v]
        v_next = input[n_q + n_v: n_q + 2 * n_v]
        tau_a = input[n_q + 2 * n_v:]

        # update MBP context.
        q_float = autoDiffToValueMatrix(q).squeeze()
        self.plant.SetPositions(self.context_plant, q_float)
        self.plant_ad.SetPositions(self.context_plant_ad,
                                   initializeAutoDiff(q_float))

        # collision query.
        signed_distance_pairs_ad = self.detect_collision_ad()
        n_c = len(signed_distance_pairs_ad)
        n_d = 2  # TODO: support 3D...
        Jn_ad = np.zeros((n_c, n_v), dtype=object)
        Jf_ad = np.zeros((n_c * n_d, n_v), dtype=object)
        phi = np.zeros(n_c)
        U = np.zeros(n_c)

        i_f_start = 0
        for i_c, sdp in enumerate(signed_distance_pairs_ad):
            # Note that sdp.id_A is always the collision geometry of the object.
            bodyA = self.plant_ad.get_body(
                self.g_id_to_mbp_body_id_map[sdp.id_A])
            bodyB = self.plant_ad.get_body(
                self.g_id_to_mbp_body_id_map[sdp.id_B])
            X_AGa = self.X_FG_dict[sdp.id_A].cast[AutoDiffXd]()
            X_BGb = self.X_FG_dict[sdp.id_B].cast[AutoDiffXd]()
            p_AcA_A = X_AGa.multiply(sdp.p_ACa)
            p_BcB_B = X_BGb.multiply(sdp.p_BCb)
            n_A_W = sdp.nhat_BA_W
            d_A_W = self.calc_tangent_vectors_yz(n_A_W)
            n_B_W = -n_A_W
            d_B_W = -d_A_W

            # update Jn and Jf
            self.q_sim.update_normal_and_tangential_jacobian_rows(
                body=bodyA, pC_D=p_AcA_A, n_W=n_A_W, d_W=d_A_W,
                i_c=i_c, n_di=n_d, i_f_start=i_f_start,
                position_indices=None,
                Jn=Jn_ad, Jf=Jf_ad,
                jacobian_wrt_variable=JacobianWrtVariable.kV,
                plant=self.plant_ad, context=self.context_plant_ad)

            self.q_sim.update_normal_and_tangential_jacobian_rows(
                body=bodyB, pC_D=p_BcB_B, n_W=n_B_W, d_W=d_B_W,
                i_c=i_c, n_di=n_d, i_f_start=i_f_start,
                position_indices=None,
                Jn=Jn_ad, Jf=Jf_ad,
                jacobian_wrt_variable=JacobianWrtVariable.kV,
                plant=self.plant_ad, context=self.context_plant_ad)

            phi[i_c] = sdp.distance.value()
            U[i_c] = \
                self.q_sim.get_friction_coefficient_for_signed_distance_pair(
                    sdp)
            i_f_start += n_d

        dq = autoDiffToGradientMatrix(q)
        Jn = autoDiffToValueMatrix(Jn_ad)
        # compute derivative of phi w.r.t q
        phi_ad = initializeAutoDiffGivenGradientMatrix(phi, Jn.dot(dq))

        # Jacobian J = Jn+ mu * Jf
        J_ad_q = Jf_ad.copy()
        phi_J_ad = np.zeros_like(J_ad_q[:, 0], dtype=object)

        i_f_start = 0
        for i_c in range(n_c):
            J_ad_q[i_f_start: i_f_start + n_d] = \
                Jn_ad[i_c] + U[i_c] * Jf_ad[i_f_start: i_f_start + n_d]
            phi_J_ad[i_f_start: i_f_start + n_d] = phi_ad[i_c]
            i_f_start += n_d

        J_ad = np.zeros_like(J_ad_q, dtype=J_ad_q.dtype)
        for i in range(J_ad.shape[0]):
            J_ad[i] = initializeAutoDiffGivenGradientMatrix(
                autoDiffToValueMatrix(J_ad_q[i]),
                autoDiffToGradientMatrix(J_ad_q[i]).dot(dq))

        # compute gravity torque
        tau_ext_u_dict = self.q_sim.calc_gravity_for_unactuated_models()

        # Compute Q(no derivative) and tau.
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context_plant)
        Q = np.zeros((n_v, n_v))
        tau = np.zeros(n_v, dtype=object)

        for model in self.q_sim.models_unactuated:
            idx_v_model = self.q_sim.velocity_indices[model]
            tau[idx_v_model] = tau_ext_u_dict[model]

            ixgrid = np.ix_(idx_v_model, idx_v_model)
            Q[ixgrid] = M[ixgrid]

        idx_i, idx_j = np.diag_indices(n_v)
        h = self.q_sim.sim_settings.time_step
        for i, model in enumerate(self.q_sim.models_actuated):
            idx_v_model = self.q_sim.velocity_indices[model]
            # tau_a = Kq_a[model].dot(dq_a_cmd).
            # TODO: this is hard-coded.
            tau[idx_v_model] = tau_a[i * 2: (i+1) * 2]

            Q[idx_i[idx_v_model], idx_j[idx_v_model]] = \
                self.q_sim.K_a[model].diagonal() * h ** 2

        phi_J_ad_next_by_h = phi_J_ad / h + J_ad.dot(v_next)
        # dynamic: Q.dot(v_next - v)
        # quasi-dynamic: Q.dot(v_next), assuming v = 0.
        nabla_f0 = Q.dot(v_next) - h * tau

        t = self.q_sim.sim_settings.log_barrier_weight
        output = t * nabla_f0 - \
                 np.sum(J_ad.T / phi_J_ad_next_by_h, axis=1)

        return output


traj_opt = TrajectoryOptimizer()

#%% collision detection tests.
name_to_model_map = traj_opt.q_sim.get_model_instance_name_to_index_map()
q0_dict = {name_to_model_map[name]: q0 for name, q0 in q0_dict_str.items()}
q0_ad = initializeAutoDiff(traj_opt.q_dict_to_vec(q0_dict))
q0_ad_dict = traj_opt.vec_to_q_dict(q0_ad)

traj_opt.update_configuration(q0_ad_dict)

signed_distance_pairs_float = traj_opt.detect_collision()
signed_distance_pairs_ad = traj_opt.detect_collision_ad()

#%%
diagram_ad = traj_opt.q_sim.diagram.ToAutoDiffXd()
plant_ad = diagram_ad.GetSubsystemByName(traj_opt.plant.get_name())
sg_ad = diagram_ad.GetSubsystemByName(traj_opt.sg.get_name())

context_ad = diagram_ad.CreateDefaultContext()
context_plant_ad = diagram_ad.GetSubsystemContext(plant_ad, context_ad)
context_sg_ad = diagram_ad.GetSubsystemContext(sg_ad, context_ad)

# Update state in self.plant_context_ad.
for model_instance_idx, q_ad in q0_ad_dict.items():
    plant_ad.SetPositions(context_plant_ad, model_instance_idx, q_ad)

query_object_ad = sg_ad.get_query_output_port().Eval(context_sg_ad)
signed_distance_pairs = [
    query_object_ad.ComputeSignedDistancePairClosestPoints(*geometry_pair)
    for geometry_pair in traj_opt.collision_pairs]

#%%
for sdp1, sdp2 in zip(signed_distance_pairs_ad, signed_distance_pairs):
    pass


#%% test hand-written collision detection.
for sdp, sdp_ad in zip(signed_distance_pairs_float, signed_distance_pairs_ad):
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


#%% call eval_dynamics_constraint
q0 = traj_opt.q_dict_to_vec(q0_dict)
v0 = np.zeros_like(q0)
v1 = np.zeros_like(v0)
tau_a0 = np.array([0, 0, 0, 0.])
input_ad = initializeAutoDiff(np.hstack([q0, v0, v1, tau_a0])).squeeze()
output = traj_opt.eval_dynamics_constraint(input_ad)

#%% initial guess from free-falling trajectory
loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    robot_info_dict=robot_info_dict,
    object_sdf_paths=object_sdf_dict,
    h=h,
    gravity=gravity,
    is_visualizing=True,
    real_time_rate=0.0,
    sim_settings=sim_settings)

#%%
n_q = traj_opt.plant.num_positions()
n_v = traj_opt.q_sim.n_v
n_a = traj_opt.plant.num_actuated_dofs()
q_traj_initial = np.zeros((T, n_q))
v_traj_initial = np.zeros((T, n_v))
tau_a_traj_initial = np.zeros((T, n_a))

for i in range(T):
    q_dict_i = {name: logger.data()[:, i]
                for name, logger in loggers_dict_quasistatic_str.items()}
    q_traj_initial[i] = traj_opt.q_dict_to_vec(q_dict_i)

for i in range(1, T):
    v_traj_initial[i] = (q_traj_initial[i] - q_traj_initial[i - 1]) / h

    # tau_a_cmd.
    model_l = name_to_model_map[robot_l_name]
    indices_l = traj_opt.q_sim.velocity_indices[model_l]
    q_a_l_cmd = q_a_traj_dict_str[robot_l_name].value(i * h).squeeze()
    q_a_l = q_traj_initial[i][indices_l]
    tau_a_cmd_l = traj_opt.q_sim.K_a[model_l].dot(q_a_l_cmd - q_a_l)

    model_r = name_to_model_map[robot_r_name]
    indices_r = traj_opt.q_sim.velocity_indices[model_r]
    q_a_r_cmd = q_a_traj_dict_str[robot_r_name].value(i * h).squeeze()
    q_a_r = q_traj_initial[i][indices_r]
    tau_a_cmd_r = traj_opt.q_sim.K_a[model_r].dot(q_a_r_cmd - q_a_r)

    tau_a_traj_initial[i, :2] = tau_a_cmd_l
    tau_a_traj_initial[i, 2:] = tau_a_cmd_r


#%%
prog = mp.MathematicalProgram()

# states.
q_traj = prog.NewContinuousVariables(T, n_q, "q")
v_traj = prog.NewContinuousVariables(T, n_v, "v")
tau_a_traj = prog.NewContinuousVariables(T, n_a, "tau_a")

# constrain initial conditions.
prog.AddLinearEqualityConstraint(q_traj[0], q0)
prog.AddLinearEqualityConstraint(v_traj[0], v0)
prog.AddLinearEqualityConstraint(tau_a_traj[0], tau_a0)


# Constraints.
for l in range(T - 1):
    # integration.
    lhs = q_traj[l + 1] - q_traj[l]
    rhs = h * v_traj[l + 1]
    for lhs_i, rhs_i in zip(lhs, rhs):
        prog.AddLinearConstraint(lhs_i == rhs_i)

    # dynamics.
    input = np.hstack([q_traj[l], v_traj[l], v_traj[l + 1], tau_a_traj[l + 1]])
    prog.AddConstraint(traj_opt.eval_dynamics_constraint,
                       lb=np.zeros(n_v), ub=np.zeros(n_v), vars=input)


# initial guess
prog.SetInitialGuess(q_traj, q_traj_initial)
prog.SetInitialGuess(v_traj, v_traj_initial)
prog.SetInitialGuess(tau_a_traj, tau_a_traj_initial)

# Cost
prog.AddQuadraticCost((tau_a_traj[1:]**2).sum())
idx_object = traj_opt.q_sim.velocity_indices[name_to_model_map[object_name]]
prog.AddQuadraticCost(10 * ((q_traj[-1][idx_object] - np.array([0, 1.4, 0]))**2).sum())

#%% Solve.
# cProfile.runctx('solver_snopt.Solve(prog)',
#                 globals=globals(), locals=locals(),
#                 filename='mp_solve_stats')
result = solver_snopt.Solve(prog)
print(result.get_solution_result())


#%%
traj_opt.plant.SetPositions(traj_opt.context_plant, q_traj_initial[-1])
sdpairs = traj_opt.detect_collision()
for sdp in sdpairs:
    print(sdp.distance)


#%%
from pydrake.all import FindResourceOrThrow