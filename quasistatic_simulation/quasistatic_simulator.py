from typing import List, Union, Dict, Tuple
from collections import namedtuple
import copy

import numpy as np
import cvxpy as cp

from pydrake.all import (QueryObject, ModelInstanceIndex, GurobiSolver,
                         AbstractValue, DiagramBuilder, MultibodyPlant,
                         SceneGraph, ExternallyAppliedSpatialForce)
from pydrake.systems.meshcat_visualizer import (ConnectMeshcatVisualizer,
    MeshcatContactVisualizer)
from pydrake.solvers import mathematicalprogram as mp
from pydrake.multibody.tree import JacobianWrtVariable, RigidBody
from pydrake.multibody.plant import (
    PointPairContactInfo, ContactResults,
    CalcContactFrictionFromSurfaceProperties)
from pydrake.geometry import PenetrationAsPointPair

from contact_aware_control.contact_particle_filter.utils_cython import (
    CalcTangentVectors)

from .environment_setup import RobotInfo, create_plant_with_robots_and_objects


class MyContactInfo(object):
    """
    Used as an intermediate storage structure for constructing
    PointPairContactInfo.
    n_W is pointing into body B.
    dC_W are the tangent vectors spanning the tangent plane at the
    contact point.
    """

    def __init__(self, bodyA_index, bodyB_index, geometry_id_A, geometry_id_B,
                 p_WC_W, n_W, dC_W):
        self.bodyA_index = bodyA_index
        self.bodyB_index = bodyB_index
        self.geometry_id_A = geometry_id_A
        self.geometry_id_B = geometry_id_B
        self.p_WC_W = p_WC_W
        self.n_W = n_W
        self.dC_W = dC_W


"""
@param is_quasi_dynamic: bool. If True, dynamics of unactauted objects is 
    given by sum(F) = M @ (v_(l+1) - 0). If False, it becomes sum(F) = 0 
    instead. 
    This is only relevant when is_constrained == False. Not having an mass 
    matrix can sometimes makes the unconstrained program unbounded. 
@param is_unconstrained: bool. If False, solves the standard QP for system 
    states at the next time step. If True, solves an unconstrained version of 
    the QP, obtained by moving inequality constraints into the objective with 
    log barrier functions. 
@param log_barrier_weight: float, used only when is_unconstrained == True. 
"""
SimulationSettings = namedtuple(
    "SimulationSettings",
    field_names=["is_quasi_dynamic", "is_unconstrained", "log_barrier_weight"],
    defaults=[False, False, 1e6])


class QuasistaticSimulator:
    def __init__(self, robot_info_dict: Dict[str, RobotInfo],
                 object_sdf_paths: Dict[str, str], gravity: np.ndarray,
                 nd_per_contact: int, sim_settings: SimulationSettings,
                 internal_vis: bool = False):
        """
        Let's assume that
        - Each rigid body has one contact geometry.
        all joints of the robot.
        :param sim_settings:
        :param robot_info_dict:
        """
        self.sim_settings = sim_settings
        # Construct diagram system for proximity queries, Jacobians.
        builder = DiagramBuilder()
        plant, scene_graph, robot_model_list, object_model_list = \
            create_plant_with_robots_and_objects(
                builder, robot_info_dict, object_sdf_paths, 1e-3, gravity)

        # visualization.
        self.internal_vis = internal_vis
        if internal_vis:
            viz = ConnectMeshcatVisualizer(
                builder, scene_graph,
                frames_to_draw={"three_link_arm": {"link_ee"}})
            # ContactVisualizer
            contact_viz = MeshcatContactVisualizer(
                meshcat_viz=viz, plant=plant)
            builder.AddSystem(contact_viz)
            self.viz = viz
            self.contact_viz = contact_viz

        diagram = builder.Build()
        self.diagram = diagram
        self.plant = plant
        self.scene_graph = scene_graph
        self.inspector = scene_graph.model_inspector()

        self.context = diagram.CreateDefaultContext()
        self.context_plant = diagram.GetMutableSubsystemContext(
            plant, self.context)
        self.context_sg = diagram.GetMutableSubsystemContext(
            scene_graph, self.context)

        # Internal visualization is used when QuasistaticSimulator is used
        # outside the Systems framework.
        if internal_vis:
            self.viz.vis.delete()
            self.viz.load()
            self.context_meshcat = diagram.GetMutableSubsystemContext(
                self.viz, self.context)
            self.context_meshcat_contact = diagram.GetMutableSubsystemContext(
                self.contact_viz, self.context)

        self.models_unactuated = object_model_list
        self.models_actuated = robot_model_list
        self.models_all = object_model_list + robot_model_list

        # body indices for each model in self.models_list.
        self.body_indices_dict = dict()
        # velocity indices (into the generalized velocity vector of the MBP)
        self.velocity_indices_dict = dict()
        self.n_v_dict = dict()
        self.n_v = plant.num_velocities()

        n_v = 0
        for model in self.models_all:
            velocity_indices = self.get_velocity_indices_for_model(model)
            self.velocity_indices_dict[model] = velocity_indices
            self.n_v_dict[model] = len(velocity_indices)
            self.body_indices_dict[model] = plant.GetBodyIndices(model)

            n_v += len(velocity_indices)

        self.nd_per_contact = nd_per_contact
        # Sanity check.
        assert plant.num_velocities() == n_v

        # stiffness matrices.
        self.Kq_a = dict()
        for i, model in enumerate(self.models_actuated):
            model_name = plant.GetModelInstanceName(model)
            joint_stiffness = robot_info_dict[model_name].joint_stiffness
            assert self.n_v_dict[model] == joint_stiffness.size
            self.Kq_a[model] = np.diag(joint_stiffness).astype(float)

        # Find planar model instances.
        # TODO: it is assumed that each unactuated model instance contains
        #  only one rigid body.
        self.is_planar_dict = dict()
        for model in self.models_unactuated:
            n_v = self.n_v_dict[model]
            n_q = plant.num_positions(model)

            if n_v == 6 and n_q == 7:
                body_indices = self.plant.GetBodyIndices(model)
                assert len(body_indices) == 1
                assert self.plant.get_body(body_indices[0]).is_floating()
                self.is_planar_dict[model] = False
            else:
                self.is_planar_dict[model] = True

        # solver
        self.solver = GurobiSolver()
        assert self.solver.available()

        '''
        Both self.contact_results and self.query_object are updated by calling
        self.step_anitescu.
        '''
        # For contact force visualization. It is updated when
        #   self.calc_contact_results is called.
        self.contact_results = ContactResults()

        # For system state visualization. It is updated when
        #   self.update_configuration is called.
        self.query_object = QueryObject()

        # Logging num of contacts and solver time.
        self.nc_log = []
        self.nd_log = []
        self.optimizer_time_log = []

    def get_robot_name_to_model_instance_dict(self):
        name_to_model = dict()
        for model in self.models_all:
            name_to_model[self.plant.GetModelInstanceName(model)] = model
        return name_to_model

    def set_sim_settings(self, new_settings: SimulationSettings):
        self.sim_settings = new_settings

    def get_model_instance_indices(self):
        return (copy.copy(self.models_unactuated),
                copy.copy(self.models_actuated))

    def update_configuration(
            self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        """
        :param q_dict: A dict of np arrays keyed by model instance indices.
            Each array is the configuration of a model instance in
            self.models_list.
        :return:
        """
        # Update state in plant_context
        assert len(q_dict) == len(self.models_all)

        for model_instance_idx, q in q_dict.items():
            self.plant.SetPositions(
                self.context_plant, model_instance_idx, q)

        # Update query object.
        self.query_object = self.scene_graph.get_query_output_port().Eval(
            self.context_sg)

    def get_current_configuration(self):
        """
        :return: a dictionary containing the current positions of all model
            instances stored in self.context_plant, keyed by
            ModelInstanceIndex.
        """
        return {model: self.plant.GetPositions(self.context_plant, model)
                for model in self.models_all}

    def draw_current_configuration(self):
        # Body poses
        self.viz.DoPublish(self.context_meshcat, [])

        # Contact forces
        self.contact_viz.GetInputPort("contact_results").FixValue(
            self.context_meshcat_contact,
            AbstractValue.Make(self.contact_results))

        self.contact_viz.DoPublish(self.context_meshcat_contact, [])

    def update_normal_and_tangential_jacobian_rows(
            self,
            body: RigidBody,
            pC_D: np.ndarray,
            n_W: np.ndarray,
            d_W: np.ndarray,
            i_c: int,
            n_di: int,
            i_f_start: int,
            position_indices: Union[List[int], None],
            Jn: np.ndarray,
            Jf: np.ndarray,
            jacobian_wrt_variable: JacobianWrtVariable):
        """
        Updates corresponding rows of Jn and Jf.
        :param body: a RigidBody object that belongs to either
            self.body_indices_actuated or self.body_indices_unactuated.
            D is the body frame of body.
        :param pC_D: contact point in frame D.
        :param n_W: contact normal pointing into body, expressed in W.
        :param d_W: tangent vectors spanning the tangent plane.
        :param i_C: contact index, the index of the row of Jn to be modified.
        :param n_di: number of tangent vectors spanning the tangent plane.
        :param i_f_start: starting row Jf to be modified.
        :param position_indices: columns of J_q corresponding to the model
            instance to which body belongs.
        :param Jn: normal jacobian of shape(n_c, len(position_indices)).
        :param Jf: tangent jacobian of shape(n_f, len(position_indices)).
        :return: None.
        """
        J_WBi = self.plant.CalcJacobianTranslationalVelocity(
            context=self.context_plant,
            with_respect_to=jacobian_wrt_variable,
            frame_B=body.body_frame(),
            p_BoBi_B=pC_D,
            frame_A=self.plant.world_frame(),
            frame_E=self.plant.world_frame())
        if position_indices is not None:
            J_WBi = J_WBi[:, position_indices]

        Jn[i_c] += n_W.dot(J_WBi)
        Jf[i_f_start: i_f_start + n_di] += d_W.dot(J_WBi)

    def find_model_instance_index_for_body(self, body):
        for model, body_indices in self.body_indices_dict.items():
            if body.index() in body_indices:
                return model

    def calc_gravity_for_unactuated_models(self):
        gravity_all = self.plant.CalcGravityGeneralizedForces(
            self.context_plant)
        return {model: gravity_all[self.velocity_indices_dict[model]]
                for model in self.models_unactuated}

    def get_generalized_force_from_external_spatial_force(
            self, easf_list: List[ExternallyAppliedSpatialForce]):
        #TODO: test this more thoroughly.
        tau_ext_actuated = {
            model: np.zeros(self.n_v_dict[model])
            for model in self.models_actuated}

        for easf in easf_list:
            body = self.plant.get_body(easf.body_index)
            model = body.model_instance()
            assert model in self.models_actuated

            F_Bq_W = easf.F_Bq_W.get_coeffs()  # [tau, force]
            J = self.plant.CalcJacobianSpatialVelocity(
                context=self.context_plant,
                with_respect_to=JacobianWrtVariable.kV,
                frame_B=body.body_frame(),
                p_BP=easf.p_BoBq_B,
                frame_A=self.plant.world_frame(),
                frame_E=self.plant.world_frame())

            tau_ext_actuated[model] += \
                J[:, self.velocity_indices_dict[model]].T.dot(F_Bq_W)

        return tau_ext_actuated

    def calc_contact_jacobians(self, contact_detection_tolerance):
        """
        For all contact detected by scene graph, computes Jn and Jf.
        q = [q_u, q_a]
        :param q:
        :return:
        """
        # Evaluate contacts.
        query_object = self.query_object
        signed_distance_pairs = \
            query_object.ComputeSignedDistancePairwiseClosestPoints(
                contact_detection_tolerance)

        n_c = len(signed_distance_pairs)
        n_d = np.full(n_c, self.nd_per_contact)
        n_f = n_d.sum()
        U = np.zeros(n_c)

        self.nc_log.append(n_c)
        self.nd_log.append(n_d.sum())

        phi = np.zeros(n_c)
        Jn = np.zeros((n_c, self.n_v))
        Jf = np.zeros((n_f, self.n_v))

        contact_info_list = []

        i_f_start = 0
        for i_c, sdp in enumerate(signed_distance_pairs):
            # print("contact %i"%i_c)
            # print(self.inspector.GetNameByGeometryId(sdp.id_A))
            # print(self.inspector.GetNameByGeometryId(sdp.id_B))
            # print(self.inspector.GetFrameId(sdp.id_A))
            # print(self.inspector.GetFrameId(sdp.id_B))
            # print("distance: ", sdp.distance)
            # print("")
            '''
            A and B denote the body frames of bodyA and bodyB.
            Fa/b is the contact geometry frame relative to the body to which
                the contact geometry belongs. sdp.p_ACa is relative to frame 
                Fa (geometry frame), not frame A (body frame). 
            p_AcA_A is the coordinates of the "contact" point C1 relative
                to the body frame A expressed in frame A.
            '''

            phi[i_c] = sdp.distance
            U[i_c] = self.get_friction_coefficient_from_signed_distance_pair(
                sdp)
            bodyA = self.get_mbp_body_from_scene_graph_geometry(sdp.id_A)
            bodyB = self.get_mbp_body_from_scene_graph_geometry(sdp.id_B)
            X_AGa = self.inspector.GetPoseInFrame(sdp.id_A)
            X_BGb = self.inspector.GetPoseInFrame(sdp.id_B)
            p_AcA_A = X_AGa.multiply(sdp.p_ACa)
            p_BcB_B = X_BGb.multiply(sdp.p_BCb)

            # TODO: it is assumed contact exists only between model
            #  instances, not between bodies within the same model instance.
            model_A = self.find_model_instance_index_for_body(bodyA)
            model_B = self.find_model_instance_index_for_body(bodyB)
            is_A_in = model_A is not None
            is_B_in = model_B is not None

            if is_A_in and is_B_in:
                '''
                When a contact pair exists between an unactuated body and
                 an actuated body, we need dC_a_W = -dC_u_W. In contrast,
                 if a contact pair contains a body that is neither actuated nor
                 unactuated, e.g. the ground, we do not need dC_a_W = -dC_u_W.
                
                As CalcTangentVectors(n, nd) != - CalcTangentVectors(-n, nd), 
                    care is needed to ensure that the conditions above are met.
                    
                The normal n_A/B_W needs to point into body A/B, respectively. 
                '''
                n_A_W = sdp.nhat_BA_W
                d_A_W = CalcTangentVectors(n_A_W, n_d[i_c])
                n_B_W = -n_A_W
                d_B_W = -d_A_W

                # new Jn and Jf
                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyA, pC_D=p_AcA_A, n_W=n_A_W, d_W=d_A_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn, Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyB, pC_D=p_BcB_B, n_W=n_B_W, d_W=d_B_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn, Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

            elif is_B_in:
                n_B_W = -sdp.nhat_BA_W
                d_B_W = CalcTangentVectors(n_B_W, n_d[i_c])

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyB, pC_D=p_BcB_B, n_W=n_B_W, d_W=d_B_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn, Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

            elif is_A_in:
                n_A_W = sdp.nhat_BA_W
                d_A_W = CalcTangentVectors(n_A_W, n_d[i_c])

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyA, pC_D=p_AcA_A, n_W=n_A_W, d_W=d_A_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn, Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV)
            else:
                # either A or B is in self.body_indices_list
                raise RuntimeError("At least one body in a contact pair "
                                   "should be in self.body_indices_list.")

            i_f_start += n_d[i_c]

            # Store contact positions in order to draw contact forces later.
            #TODO: contact forces at step (l+1) is drawn with the
            # configuration at step l.
            if is_A_in:
                X_WD = self.plant.EvalBodyPoseInWorld(
                    self.context_plant, bodyA)
                contact_info_list.append(
                    MyContactInfo(
                        bodyA_index=bodyB.index(),
                        bodyB_index=bodyA.index(),
                        geometry_id_A=sdp.id_B,
                        geometry_id_B=sdp.id_A,
                        p_WC_W=X_WD.multiply(p_AcA_A),
                        n_W=n_A_W,
                        dC_W=d_A_W))
            elif is_B_in:
                X_WD = self.plant.EvalBodyPoseInWorld(
                    self.context_plant, bodyB)
                contact_info_list.append(
                    MyContactInfo(
                        bodyA_index=bodyA.index(),
                        bodyB_index=bodyB.index(),
                        geometry_id_A=sdp.id_A,
                        geometry_id_B=sdp.id_B,
                        p_WC_W=X_WD.multiply(p_BcB_B),
                        n_W=n_B_W,
                        dC_W=d_B_W))
            else:
                raise RuntimeError("At least one body in a contact pair "
                                   "should be unactuated.")

        return n_c, n_d, n_f, Jn, Jf, phi, U, contact_info_list

    def update_contact_results(self, my_contact_info_list: List[MyContactInfo],
                               beta: np.ndarray, h: float, n_c: int,
                               n_d: np.ndarray,
                               mu_list: np.ndarray):
        assert len(my_contact_info_list) == n_c
        contact_results = ContactResults()
        i_f_start = 0
        for i_c, my_contact_info in enumerate(my_contact_info_list):
            i_f_end = i_f_start + n_d[i_c]
            beta_i = beta[i_f_start: i_f_end]
            f_normal_W = my_contact_info.n_W * beta_i.sum() / h
            f_tangential_W = \
                my_contact_info.dC_W.T.dot(beta_i) * mu_list[i_c] / h
            point_pair = PenetrationAsPointPair()
            point_pair.id_A = my_contact_info.geometry_id_A
            point_pair.id_B = my_contact_info.geometry_id_B
            contact_results.AddContactInfo(
                PointPairContactInfo(
                    my_contact_info.bodyA_index,
                    my_contact_info.bodyB_index,
                    f_normal_W + f_tangential_W,
                    my_contact_info.p_WC_W,
                    0, 0, point_pair))

            i_f_start += n_d[i_c]

        self.contact_results = contact_results

    def get_mbp_body_from_scene_graph_geometry(self, g_id):
        f_id = self.inspector.GetFrameId(g_id)
        return self.plant.GetBodyFromFrameId(f_id)

    def get_position_indices_for_model(self, model_instance_index):
        selector = np.arange(self.plant.num_positions())
        return self.plant.GetPositionsFromArray(
            model_instance_index, selector).astype(np.int)

    def get_velocity_indices_for_model(self, model_instance_index):
        selector = np.arange(self.plant.num_velocities())
        return self.plant.GetVelocitiesFromArray(
            model_instance_index, selector).astype(np.int)

    def get_friction_coefficient_from_signed_distance_pair(self, sdp):
        props_A = self.inspector.GetProximityProperties(sdp.id_A)
        props_B = self.inspector.GetProximityProperties(sdp.id_B)
        cf_A = props_A.GetProperty("material", "coulomb_friction")
        cf_B = props_B.GetProperty("material", "coulomb_friction")
        cf = CalcContactFrictionFromSurfaceProperties(cf_A, cf_B)
        return cf.static_friction()

    def calc_jacobian_and_phi(self, contact_detection_tolerance):
        (n_c, n_d, n_f, Jn, Jf, phi_l, U,
         contact_info_list) = self.calc_contact_jacobians(
            contact_detection_tolerance)

        phi_constraints = np.zeros(n_f)
        J = np.zeros_like(Jf)
        j_start = 0
        for i_c in range(n_c):
            for j in range(n_d[i_c]):
                idx = j_start + j
                J[idx] = Jn[i_c] + U[i_c] * Jf[idx]
                phi_constraints[idx] = phi_l[i_c]
            j_start += n_d[i_c]

        return phi_constraints, J, contact_info_list, n_c, n_d, n_f, U

    def step_qp(self,
                q_dict: Dict[ModelInstanceIndex, np.ndarray],
                q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
                tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
                h: float,
                phi_constraints: np.ndarray,
                J: np.ndarray):

        prog = mp.MathematicalProgram()
        # generalized velocity times time step.
        vh = prog.NewContinuousVariables(self.n_v, "v_h")
        v_h_dict = dict()
        for model in self.models_all:
            v_h_dict[model] = vh[self.velocity_indices_dict[model]]

        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context_plant)
        for model in self.models_unactuated:
            idx_v_model = self.velocity_indices_dict[model]
            ixgrid = np.ix_(idx_v_model, idx_v_model)

            if self.sim_settings.is_quasi_dynamic:
                prog.AddQuadraticCost(M[ixgrid] / h,
                                      -tau_ext_dict[model] * h,
                                      v_h_dict[model])
            else:
                prog.AddLinearCost(
                    -tau_ext_dict[model] * h, 0, v_h_dict[model])

        for model in self.models_actuated:
            dq_a_cmd = q_a_cmd_dict[model] - q_dict[model]
            tau_a = self.Kq_a[model].dot(dq_a_cmd) + tau_ext_dict[model]
            prog.AddQuadraticCost(self.Kq_a[model] * h,
                                  -tau_a * h,
                                  v_h_dict[model])

        constraints = prog.AddLinearConstraint(
            J, -phi_constraints, np.full_like(phi_constraints, np.inf), vh)

        result = self.solver.Solve(prog, None, None)
        # self.optimizer_time_log.append(
        #     result.get_solver_details().optimizer_time)
        assert result.get_solution_result() == mp.SolutionResult.kSolutionFound
        beta = result.GetDualSolution(constraints)
        beta = np.array(beta).squeeze()

        v_h_value_dict = dict()
        for model in v_h_dict.keys():
            v_h_value_dict[model] = result.GetSolution(v_h_dict[model])

        return v_h_value_dict, beta

    def step_unconstrained(self,
                           q_dict: Dict[ModelInstanceIndex, np.ndarray],
                           q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
                           tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
                           h: float,
                           phi_constraints: np.ndarray,
                           J: np.ndarray):
        vh = cp.Variable(self.n_v)
        v_h_dict = dict()
        for model in self.models_all:
            v_h_dict[model] = vh[self.velocity_indices_dict[model]]

        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context_plant)
        Q = np.zeros((self.n_v, self.n_v))
        idx_i, idx_j = np.diag_indices(self.n_v)
        tau_h = np.zeros(self.n_v)
        for model in self.models_unactuated:
            idx_v_model = self.velocity_indices_dict[model]
            tau_h[idx_v_model] = tau_ext_dict[model] * h

            ixgrid = np.ix_(idx_v_model, idx_v_model)
            Q[ixgrid] = M[ixgrid] / h

        for model in self.models_actuated:
            idx_v_model = self.velocity_indices_dict[model]
            dq_a_cmd = q_a_cmd_dict[model] - q_dict[model]
            tau_a = self.Kq_a[model].dot(dq_a_cmd) + tau_ext_dict[model]
            tau_h[idx_v_model] = tau_a * h

            Q[idx_i[idx_v_model], idx_j[idx_v_model]] = \
                self.Kq_a[model].diagonal() * h

        t = self.sim_settings.log_barrier_weight
        log_barriers_sum = 0.
        if len(phi_constraints) > 0:
            log_barriers_sum = cp.sum(cp.log(phi_constraints + J @ vh))
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(vh, Q) - tau_h @ vh -
                        log_barriers_sum / t))

        prob.solve()
        assert prob.status == "optimal"

        v_h_value_dict = dict()
        for model in v_h_dict.keys():
            v_h_value_dict[model] = v_h_dict[model].value

        return v_h_value_dict, np.zeros_like(phi_constraints)

    def step(self, q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
             tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
             h: float, contact_detection_tolerance: float):
        """
        This function does the following:
        1. Extracts q_dict, a dictionary containing current system
            configuration, from self.plant_context.
        2. Runs collision query by calling self.calc_contact_jacobians.
        3. Constructs and solves the quasistatic QP described in the paper.
        4. Integrates q_dict to the next time step.
        5. Calls self.update_configuration with the new q_dict.
            self.update_configuration updates self.context_plant and
            self.query_object.
        6. Updates self.contact_results.
        :param q_a_cmd_dict:
        :param tau_ext_dict:
        :param h: simulation time step.
        :param contact_detection_tolerance:
        :return: system configuration at the next time step, stored in a
            dictionary keyed by ModelInstanceIndex.
        """
        q_dict = self.get_current_configuration()
        self.update_configuration(q_dict)
        phi_constraints, J, contact_info_list, n_c, n_d, n_f, U = \
            self.calc_jacobian_and_phi(contact_detection_tolerance)

        if self.sim_settings.is_unconstrained:
            v_h_value_dict, beta = \
                self.step_unconstrained(q_dict, q_a_cmd_dict, tau_ext_dict,
                             h, phi_constraints, J)
        else:
            v_h_value_dict, beta = \
                self.step_qp(q_dict, q_a_cmd_dict, tau_ext_dict,
                             h, phi_constraints, J)

        dq_dict = dict()
        for model in self.models_actuated:
            v_h_value = v_h_value_dict[model]
            dq_dict[model] = v_h_value

        for model in self.models_unactuated:
            v_h_value = v_h_value_dict[model]
            if self.is_planar_dict[model]:
                dq_dict[model] = v_h_value
            else:
                q_u = q_dict[model]
                Q = q_u[:4]  # Quaternion Q_WB
                E = np.array([[-Q[1], Q[0], -Q[3], Q[2]],
                              [-Q[2], Q[3], Q[0], -Q[1]],
                              [-Q[3], -Q[2], Q[1], Q[0]]])

                dq_u = np.zeros(7)
                dq_u[:4] = 0.5 * E.T.dot(v_h_value[:3])
                dq_u[4:] = v_h_value[3:]
                dq_dict[model] = dq_u

        self.step_configuration(q_dict, dq_dict)
        self.update_configuration(q_dict)
        self.update_contact_results(contact_info_list, beta, h, n_c, n_d, U)
        return q_dict

    def step_configuration(self, q_dict: Dict[ModelInstanceIndex, np.ndarray],
                           dq_dict: Dict[ModelInstanceIndex, np.ndarray]):
        """
        Adds the delta of each model state, e.g. dq_u_list[i], to the
            corresponding model configuration in q_list. If q_list[i]
            includes a quaternion, the quaternion (usually the first four
            numbers of a seven-number array) is normalized.
        :param q_dict:
        :param dq_u_dict:
        :param dq_a_dict:
        :return: None.
        """
        for model in self.models_unactuated:
            q_u = q_dict[model]
            q_u += dq_dict[model]

            if self.is_planar_dict[model]:
                q_u[:4] / np.linalg.norm(q_u[:4])  # normalize quaternion

        for model in self.models_actuated:
            q_dict[model] += dq_dict[model]

    def print_sim_statcs(self):
        solver_time = np.array(self.optimizer_time_log)
        n_c_s = np.array(self.nc_log)
        n_d_s = np.array(self.nd_log)

        print("Average solver time: ", solver_time.mean())
        print("Solver time std: ", solver_time.std())
        print("Average num. contacts: ", n_c_s.mean())
        print("Average num. constraints: ", n_d_s.mean())
