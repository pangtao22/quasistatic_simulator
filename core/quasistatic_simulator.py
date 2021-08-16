from typing import List, Union, Dict, Tuple
from collections import namedtuple
import copy
import warnings
import sys

import numpy as np
import cvxpy as cp

from pydrake.all import (QueryObject, ModelInstanceIndex, GurobiSolver,
                         AbstractValue, DiagramBuilder, MultibodyPlant,
                         ExternallyAppliedSpatialForce, Context,
                         JacobianWrtVariable, RigidBody,
                         PenetrationAsPointPair, ConnectMeshcatVisualizer,
                         MeshcatContactVisualizer)
from pydrake.solvers import mathematicalprogram as mp
from pydrake.multibody.plant import (
    PointPairContactInfo, ContactResults,
    CalcContactFrictionFromSurfaceProperties)

from robotics_utilities.qp_derivatives.qp_derivatives import (
    QpDerivativesKktPinv)

from .utils import create_plant_with_robots_and_objects, calc_tangent_vectors


class MyContactInfo:
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
:param contact_detection_tolerance: Signed distance pairs whose distances are 
    greater than this value are ignored in the simulator's non-penetration 
    constraints. Unit is in meters.
:param is_quasi_dynamic: bool. If True, dynamics of unactauted objects is 
    given by sum(F) = M @ (v_(l+1) - 0). If False, it becomes sum(F) = 0 
    instead. 
    The mass matrix for unactuated objects is always added when the 
    unconstrained version of the problem is solved. Not having a mass 
    matrix can sometimes makes the unconstrained program unbounded. 
:param mode: Union['qp_mp', 'qp_cvx', 'unconstrained']. 
    - 'qp_mp': solves the standard QP for system states at the next time 
        step, using MathematicalProgram. 
    - 'qp_mp': solves the standard QP using cvxpy.
    - 'unconstrained': solves an unconstrained version of the QP, obtained by 
        moving inequality constraints into the objective with 
        log barrier functions. 
:param log_barrier_weight: float, used only when is_unconstrained == True.
:param nd_per_contact: int, number of extreme rays per contact point.
:param requires_grad: whether the gradient of v_next w.r.t the parameters of 
    the QP are computed. 
"""
field_names=[
        "gravity", "nd_per_contact", "contact_detection_tolerance",
        "is_quasi_dynamic", "mode", "log_barrier_weight", "requires_grad"
        ]
defaults=[np.array([0, 0, -9.81]), 4, 0.01,
          False, "qp_mp", 1e4, False]

if sys.version_info >= (3, 7):
    QuasistaticSimParameters = namedtuple(
        "QuasistaticSimParameters",
        field_names=field_names,
        defaults=defaults)
else:
    QuasistaticSimParameters = namedtuple(
        "QuasistaticSimParameters",
        field_names=field_names)
    QuasistaticSimParameters.__new__.__defaults__ = tuple(defaults)
    QuasistaticSimParameters = QuasistaticSimParameters

class QuasistaticSimulator:
    def __init__(self, model_directive_path: str,
                 robot_stiffness_dict: Dict[str, np.ndarray],
                 object_sdf_paths: Dict[str, str],
                 sim_params: QuasistaticSimParameters,
                 internal_vis: bool = False):
        """
        Assumptions:
        - Each rigid body has one contact geometry.
        :param robot_stiffness_dict: key: model instance name; value: 1D
            array of the stiffness of each joint in the model.
        :param object_sdf_paths: key: object model instance name; value:
            object sdf path.
        """
        self.sim_params = sim_params
        # Construct diagram system for proximity queries, Jacobians.
        builder = DiagramBuilder()
        plant, scene_graph, robot_models, object_models = \
            create_plant_with_robots_and_objects(
                builder=builder,
                model_directive_path=model_directive_path,
                robot_names=[name for name in robot_stiffness_dict.keys()],
                object_sdf_paths=object_sdf_paths,
                time_step=1e-3,  # Only useful for MBP simulations.
                gravity=sim_params.gravity)

        # visualization.
        self.internal_vis = internal_vis
        if internal_vis:
            viz = ConnectMeshcatVisualizer(builder, scene_graph)
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

        self.models_unactuated = object_models
        self.models_actuated = robot_models
        self.models_all = object_models.union(robot_models)

        # body indices for each model in self.models_list.
        self.body_indices_ = dict()
        # velocity indices (into the generalized velocity vector of the MBP)
        self.velocity_indices_ = dict()
        self.n_v_dict = dict()
        self.n_v = plant.num_velocities()

        n_v = 0
        for model in self.models_all:
            velocity_indices = self.get_velocity_indices_for_model(model)
            self.velocity_indices_[model] = velocity_indices
            self.n_v_dict[model] = len(velocity_indices)
            self.body_indices_[model] = plant.GetBodyIndices(model)

            n_v += len(velocity_indices)

        self.nd_per_contact = sim_params.nd_per_contact
        # Sanity check.
        assert plant.num_velocities() == n_v

        # stiffness matrices.
        self.Kq_a = dict()
        for i, model in enumerate(self.models_actuated):
            model_name = plant.GetModelInstanceName(model)
            joint_stiffness = robot_stiffness_dict[model_name]
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

        # step function dictionary
        self.step_function_dict = {
            'qp_mp': self.step_qp_mp,
            'qp_cvx': self.step_qp_cvx,
            'unconstrained': self.step_unconstrained}

        '''
        Both self.contact_results and self.query_object are updated by calling
        self.step(...)
        '''
        # For contact force visualization. It is updated when
        #   self.calc_contact_results is called.
        self.contact_results = ContactResults()

        # For system state visualization. It is updated when
        #   self.update_configuration is called.
        self.query_object = QueryObject()

        # gradients
        self.Dv_nextDb = None
        self.Dv_nextDe = None
        self.Dq_nextDq = None
        self.Dq_nextDqa_cmd = None
        self.dqp_kkt = QpDerivativesKktPinv()

        # TODO: these are not used right now.
        # Logging num of contacts and solver time.
        self.nc_log = []
        self.nd_log = []
        self.optimizer_time_log = []

    def get_plant(self):
        return self.plant

    def get_scene_graph(self):
        return self.scene_graph

    def get_all_models(self):
        return self.models_all

    def get_actuated_models(self):
        return self.models_actuated

    def get_Dq_nextDq(self):
        return self.Dq_nextDq

    def get_Dq_nextDqa_cmd(self):
        return self.Dq_nextDqa_cmd

    def get_positions(self, model: ModelInstanceIndex):
        return self.plant.GetPositions(self.context_plant, model)

    def get_query_object(self):
        return self.query_object

    def get_contact_results(self):
        return self.contact_results

    def get_velocity_indices(self):
        return self.velocity_indices_

    def num_actuated_dofs(self):
        return np.sum(
            [self.n_v_dict[model] for model in self.models_actuated])

    def num_unactuated_dof(self):
        return np.sum(
            [self.n_v_dict[model] for model in self.models_unactuated])

    def get_dynamics_derivatives(self):
        return (np.copy(self.Dv_nextDb), np.copy(self.Dv_nextDe),
                np.copy(self.Dq_nextDq), np.copy(self.Dq_nextDqa_cmd))

    def get_robot_name_to_model_instance_dict(self):
        name_to_model = dict()
        for model in self.models_all:
            name_to_model[self.plant.GetModelInstanceName(model)] = model
        return name_to_model

    def update_mbp_positions(
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

    def get_mbp_positions(self):
        """
        :return: a dictionary containing the current positions of all model
            instances stored in self.context_plant, keyed by
            ModelInstanceIndex.
        """
        return {model: self.plant.GetPositions(self.context_plant, model)
                for model in self.models_all}

    def draw_current_configuration(self, draw_forces=True):
        # Body poses
        self.viz.DoPublish(self.context_meshcat, [])

        # Contact forces
        if draw_forces:
            self.contact_viz.GetInputPort("contact_results").FixValue(
                self.context_meshcat_contact,
                AbstractValue.Make(self.contact_results))
            self.contact_viz.DoPublish(self.context_meshcat_contact, [])

    def calc_tau_ext(self, easf_list: List[ExternallyAppliedSpatialForce]):
        """
        Combines tau_a_ext from
            self.get_generalized_force_from_external_spatial_force
         and tau_u_ext from
            self.calc_gravity_for_unactuated_models.
        """

        # Gravity for unactuated models.
        tau_ext_u_dict = self.calc_gravity_for_unactuated_models()

        # external spatial force for actuated models.
        tau_ext_a_dict = \
            self.get_generalized_force_from_external_spatial_force(
                easf_list)
        return {**tau_ext_a_dict, **tau_ext_u_dict}

    def animate_system_trajectory(self, h: float,
            q_dict_traj: List[Dict[ModelInstanceIndex, np.ndarray]]):
        self.viz.draw_period = h
        self.viz.reset_recording()
        self.viz.start_recording()
        for q_dict in q_dict_traj:
            self.update_mbp_positions(q_dict)
            self.draw_current_configuration(draw_forces=False)

        self.viz.stop_recording()
        self.viz.publish_recording()

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
            jacobian_wrt_variable: JacobianWrtVariable,
            plant: MultibodyPlant = None,
            context: Context = None):
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
        :param plant: if None, use self.plant.
        :return: None.
        """
        if plant is None:
            plant = self.plant
            context = self.context_plant
        J_WBi = plant.CalcJacobianTranslationalVelocity(
            context=context,
            with_respect_to=jacobian_wrt_variable,
            frame_B=body.body_frame(),
            p_BoBi_B=pC_D,
            frame_A=plant.world_frame(),
            frame_E=plant.world_frame())
        if position_indices is not None:
            J_WBi = J_WBi[:, position_indices]

        Jn[i_c] += n_W.dot(J_WBi)
        Jf[i_f_start: i_f_start + n_di] += d_W.dot(J_WBi)

    def find_model_instance_index_for_body(self, body):
        for model, body_indices in self.body_indices_.items():
            if body.index() in body_indices:
                return model

    def calc_gravity_for_unactuated_models(self):
        gravity_all = self.plant.CalcGravityGeneralizedForces(
            self.context_plant)
        return {model: gravity_all[self.velocity_indices_[model]]
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
                J[:, self.velocity_indices_[model]].T.dot(F_Bq_W)

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
            p_ACa_A is the coordinates of the "contact" point Ca relative
                to the body frame A expressed in frame A.
            '''

            phi[i_c] = sdp.distance
            U[i_c] = self.get_friction_coefficient_for_signed_distance_pair(
                sdp)
            bodyA = self.get_mbp_body_from_scene_graph_geometry(sdp.id_A)
            bodyB = self.get_mbp_body_from_scene_graph_geometry(sdp.id_B)
            X_AGa = self.inspector.GetPoseInFrame(sdp.id_A)
            X_BGb = self.inspector.GetPoseInFrame(sdp.id_B)
            p_ACa_A = X_AGa.multiply(sdp.p_ACa)
            p_BCb_B = X_BGb.multiply(sdp.p_BCb)

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
                d_A_W = calc_tangent_vectors(n_A_W, n_d[i_c])
                n_B_W = -n_A_W
                d_B_W = -d_A_W

                # new Jn and Jf
                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyA, pC_D=p_ACa_A, n_W=n_A_W, d_W=d_A_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn, Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyB, pC_D=p_BCb_B, n_W=n_B_W, d_W=d_B_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn, Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

            elif is_B_in:
                n_B_W = -sdp.nhat_BA_W
                d_B_W = calc_tangent_vectors(n_B_W, n_d[i_c])

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyB, pC_D=p_BCb_B, n_W=n_B_W, d_W=d_B_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn, Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

            elif is_A_in:
                n_A_W = sdp.nhat_BA_W
                d_A_W = calc_tangent_vectors(n_A_W, n_d[i_c])

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyA, pC_D=p_ACa_A, n_W=n_A_W, d_W=d_A_W,
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
                        p_WC_W=X_WD.multiply(p_ACa_A),
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
                        p_WC_W=X_WD.multiply(p_BCb_B),
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
            model_instance_index, selector).astype(int)

    def get_velocity_indices_for_model(self, model_instance_index):
        selector = np.arange(self.plant.num_velocities())
        return self.plant.GetVelocitiesFromArray(
            model_instance_index, selector).astype(int)

    def get_friction_coefficient_for_signed_distance_pair(self, sdp):
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

        return (phi_constraints, J, phi_l, Jn, contact_info_list, n_c, n_d,
                n_f, U)

    @staticmethod
    def check_cvx_status(status: str):
        if status != "optimal":
            if status == "optimal_inaccurate":
                warnings.warn("CVX solver is inaccurate.")
            else:
                raise RuntimeError(
                    "CVX solver status is {}".format(status))

    def form_Q_and_tau_h(
            self,
            q_dict: Dict[ModelInstanceIndex, np.ndarray],
            q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
            tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
            h: float):
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context_plant)
        Q = np.zeros((self.n_v, self.n_v))
        tau_h = np.zeros(self.n_v)
        for model in self.models_unactuated:
            idx_v_model = self.velocity_indices_[model]
            tau_h[idx_v_model] = tau_ext_dict[model] * h

            if self.sim_params.is_quasi_dynamic:
                ixgrid = np.ix_(idx_v_model, idx_v_model)
                Q[ixgrid] = M[ixgrid]

        idx_i, idx_j = np.diag_indices(self.n_v)
        for model in self.models_actuated:
            idx_v_model = self.velocity_indices_[model]
            dq_a_cmd = q_a_cmd_dict[model] - q_dict[model]
            tau_a = self.Kq_a[model].dot(dq_a_cmd) + tau_ext_dict[model]
            tau_h[idx_v_model] = tau_a * h

            Q[idx_i[idx_v_model], idx_j[idx_v_model]] = \
                self.Kq_a[model].diagonal() * h**2
        return Q, tau_h

    def step_qp_mp(self,
                   q_dict: Dict[ModelInstanceIndex, np.ndarray],
                   q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
                   tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
                   h: float,
                   phi_constraints: np.ndarray,
                   J: np.ndarray,
                   requires_grad: bool):
        Q, tau_h = self.form_Q_and_tau_h(
            q_dict, q_a_cmd_dict, tau_ext_dict, h)

        prog = mp.MathematicalProgram()
        # generalized velocity times time step.
        v = prog.NewContinuousVariables(self.n_v, "v")

        prog.AddQuadraticCost(Q, -tau_h, v)
        e = phi_constraints / h
        constraints = prog.AddLinearConstraint(
            A=-J,
            lb=np.full_like(phi_constraints, -np.inf),
            ub=e,
            vars=v)

        result = self.solver.Solve(prog, None, None)
        # self.optimizer_time_log.append(
        #     result.get_solver_details().optimizer_time)
        assert result.is_success()
        beta = -result.GetDualSolution(constraints)
        beta = np.array(beta).squeeze()

        # extract v_h from vector into a dictionary.
        v_values = result.GetSolution(v)
        v_h_value_dict = dict()
        for model in self.models_all:
            indices = self.velocity_indices_[model]
            v_h_value_dict[model] = v_values[indices] * h

        # compute DvDb and DvDe
        DvDb, DvDe = None, None
        if requires_grad:
            self.dqp_kkt.update_problem(
                Q=Q, b=-tau_h, G=-J, e=e, z_star=v_values, lambda_star=beta)
            DvDb = self.dqp_kkt.calc_DzDb()
            DvDe = self.dqp_kkt.calc_DzDe()

        return v_h_value_dict, beta, DvDb, DvDe

    def step_qp_cvx(self,
                    q_dict: Dict[ModelInstanceIndex, np.ndarray],
                    q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
                    tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
                    h: float,
                    phi_constraints: np.ndarray,
                    J: np.ndarray,
                    requires_grad: bool):
        Q, tau_h = self.form_Q_and_tau_h(
            q_dict, q_a_cmd_dict, tau_ext_dict, h)

        # Make a CVX problem.
        # The cholesky decomposition is needed because cp.sum_squares() is the
        # only way I've found so far to ensure the problem is DCP (
        # disciplined convex program).
        '''
        The original non-penetration constraint is given by 
            phi_constraints / h + J @ v >= 0
        The "standard form" in Cotler's paper is
            min. 1 / 2 * z.dot(Q).dot(z) + b.dot(z)
            s.t. G @ z <= e (or e >= G @ z).
        Rearranging the non-penetration constraint as:
            phi_constraints / h >= -J @ v 
        gives
            G := -J; e := phi_constraints / h.

        Objective: b := -tau_h.
        '''
        L = np.linalg.cholesky(Q)
        L_cp = cp.Parameter(L.shape)
        L_cp.value = L

        b_cp = cp.Parameter(self.n_v)
        b_cp.value = -tau_h

        n_e = len(phi_constraints)
        e_cp = cp.Parameter(n_e)
        e_cp.value = phi_constraints / h

        v = cp.Variable(self.n_v)

        constraints = [e_cp + J @ v >= 0]
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(L_cp.T @ v) + b_cp @ v),
            constraints)

        prob.solve(requires_grad=requires_grad)
        self.check_cvx_status(prob.status)

        # extract v_h from vector into a dictionary.
        v_h_value_dict = dict()
        for model in self.models_all:
            indices = self.velocity_indices_[model]
            v_h_value_dict[model] = v.value[indices] * h

        # compute DvDb and DvDe
        DvDb, DvDe = None, None
        if requires_grad:
            DvDb = np.zeros((self.n_v, self.n_v))
            DvDe = np.zeros((self.n_v, n_e))
            for i in range(self.n_v):
                dv = np.zeros(self.n_v)
                dv[i] = 1
                v.gradient = dv
                prob.backward()

                DvDb[i] = b_cp.gradient
                DvDe[i] = e_cp.gradient

            self.dqp_kkt.update_problem(
                Q=Q, b=-tau_h, G=-J, e=e_cp.value, z_star=v.value,
                lambda_star=constraints[0].dual_value)
            DvDb = self.dqp_kkt.calc_DzDb()
            DvDe = self.dqp_kkt.calc_DzDe()

        return v_h_value_dict, np.zeros_like(phi_constraints), DvDb, DvDe

    def step_unconstrained(self,
                           q_dict: Dict[ModelInstanceIndex, np.ndarray],
                           q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
                           tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
                           h: float,
                           phi_constraints: np.ndarray,
                           J: np.ndarray,
                           requires_grad: bool):
        Q, tau_h = self.form_Q_and_tau_h(
            q_dict, q_a_cmd_dict, tau_ext_dict, h)

        v = cp.Variable(self.n_v)

        t = self.sim_params.log_barrier_weight
        log_barriers_sum = 0.
        if len(phi_constraints) > 0:
            log_barriers_sum = cp.sum(cp.log(phi_constraints / h + J @ v))
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(v, Q) - tau_h @ v -
                        log_barriers_sum / t))

        prob.solve()
        self.check_cvx_status(prob.status)

        # extract v_h from vector into a dictionary.
        v_h_value_dict = dict()
        for model in self.models_all:
            indices = self.velocity_indices_[model]
            v_h_value_dict[model] = v.value[indices] * h

        # TODO: gradient not supported yet.
        DvDb, DvDe = None, None
        return v_h_value_dict, np.zeros_like(phi_constraints), DvDb, DvDe

    def step(self,
             q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
             tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
             h: float,
             mode: str,
             requires_grad: bool):
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
         :param mode: one of {'qp_mp', 'qp_cvx', 'unconstrained'}.
         :param requires_grad: whether gradient of the dynamics is computed.
         :return: system configuration at the next time step, stored in a
             dictionary keyed by ModelInstanceIndex.
         """
        q_dict = self.get_mbp_positions()

        phi_constraints, J, phi_l, Jn, contact_info_list, n_c, n_d, n_f, U = \
            self.calc_jacobian_and_phi(
                self.sim_params.contact_detection_tolerance)

        v_h_value_dict, beta, Dv_nextDb, Dv_nextDe = \
            self.step_function_dict[mode](
                q_dict, q_a_cmd_dict, tau_ext_dict, h, phi_constraints, J,
                requires_grad=requires_grad)

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
        self.update_mbp_positions(q_dict)
        self.update_contact_results(contact_info_list, beta, h, n_c, n_d, U)

        if not requires_grad:
            return q_dict

        # Gradients.
        '''
        Generic dynamical system: x_next = f(x, u). We need DfDx and DfDu.

        In a quasistatic system, x := [qu, qa], u = qa_cmd.
        q_next = q + h * E @ v_next
        v_next = argmin_{v} {0.5 * v @ Q @ v + b @ v | G @ v <= e}
            - E is not identity if rotation is represented by quaternions.
            - Dv_nextDb and Dv_nextDe are returned by self.step_*(...).

        D(q_next)Dq = I + h * E @ D(v_next)Dq
        D(q_next)D(qa_cmd) = h * E @ D(v_next)D(qa_cmd)

        Dv_nextDq = Dv_nextDb @ DbDq
                    + Dv_nextDe @ (1 / h * Dphi_constraints_Dq)
        Dv_nextDqa_cmd = Dv_nextDb @ DbDqa_cmd,
            - where DbDqa_cmd != np.vstack([0, Kq]).        
        '''
        # TODO: for now it is assumed that n_q == n_v.
        #  Dphi_constraints_Dv is used for Dphi_constraints_Dq.
        self.Dv_nextDb = Dv_nextDb
        self.Dv_nextDe = Dv_nextDe

        Dphi_constraints_Dq = np.zeros((n_f, self.n_v))
        j_start = 0
        for i_c in range(n_c):
            Dphi_constraints_Dq[j_start: j_start + n_d[i_c]] = Jn[i_c]
            j_start += n_d[i_c]

        DbDq = np.zeros((self.n_v, self.n_v))
        DbDqa_cmd = np.zeros((self.n_v, self.num_actuated_dofs()))
        j_start = 0
        for model in self.models_actuated:
            idx_v_model = self.velocity_indices_[model]
            n_v_i = len(idx_v_model)
            idx_rows = idx_v_model[:, None]
            idx_cols = np.arange(j_start, j_start + n_v_i)[None, :]
            DbDqa_cmd[idx_rows, idx_cols] = -h * self.Kq_a[model]

            idx_cols_2 = idx_v_model[None, :]
            DbDq[idx_rows, idx_cols_2] = h * self.Kq_a[model]

            j_start += n_v_i

        # Dq_nextDq
        Dv_nextDq_1 = Dv_nextDb @ DbDq
        Dv_nextDq_2 = Dv_nextDe @ Dphi_constraints_Dq / h
        Dv_nextDq = Dv_nextDq_1 + Dv_nextDq_2
        self.Dq_nextDq = np.eye(self.n_v) + h * Dv_nextDq

        # Dq_nextDqa_cmd
        self.Dq_nextDqa_cmd = h * Dv_nextDb @ DbDqa_cmd
        return q_dict

    def step_default(self,
                     q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
                     tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
                     h: float):

        return self.step(
            q_a_cmd_dict=q_a_cmd_dict,
            tau_ext_dict=tau_ext_dict,
            h=h,
            mode=self.sim_params.mode,
            requires_grad=self.sim_params.requires_grad)

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
