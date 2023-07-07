import copy
import warnings
from enum import IntEnum
from typing import List, Union, Dict

import cvxpy as cp
import numpy as np
from pydrake.all import (
    QueryObject,
    ModelInstanceIndex,
    GurobiSolver,
    MosekSolver,
    SolverOptions,
    AbstractValue,
    ExternallyAppliedSpatialForce,
    Context,
    JacobianWrtVariable,
    RigidBody,
    PenetrationAsPointPair,
    MeshcatVisualizer,
    ContactVisualizer,
    StartMeshcat,
    OsqpSolver,
    ScsSolver,
)
from pydrake.multibody.parsing import (
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
)
from pydrake.multibody.plant import (
    PointPairContactInfo,
    ContactResults,
    CalcContactFrictionFromSurfaceProperties,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
)
from pydrake.solvers import MathematicalProgram
from pydrake.systems.framework import DiagramBuilder
from qsim_cpp import (
    GradientMode,
    QuasistaticSimParameters,
    ForwardDynamicsMode,
    QpLogBarrierSolver,
)
from robotics_utilities.qp_derivatives.qp_derivatives import (
    QpDerivativesKktActive,
)

from qsim.model_paths import add_package_paths_local
from .utils import calc_tangent_vectors, is_mosek_gurobi_available

from .meshcat_visualizer_old import (
    ConnectMeshcatVisualizer as ConnectMeshcatVisualizerPy,
)


class MyContactInfo:
    """
    Used as an intermediate storage structure for constructing
    PointPairContactInfo.
    n_W is pointing into body B.
    dC_W are the tangent vectors spanning the tangent plane at the
    contact point.
    """

    def __init__(
        self,
        bodyA_index,
        bodyB_index,
        geometry_id_A,
        geometry_id_B,
        p_WC_W,
        n_W,
        dC_W,
    ):
        self.bodyA_index = bodyA_index
        self.bodyB_index = bodyB_index
        self.geometry_id_A = geometry_id_A
        self.geometry_id_B = geometry_id_B
        self.p_WC_W = p_WC_W
        self.n_W = n_W
        self.dC_W = dC_W


class InternalVisualizationType(IntEnum):
    """
    We need to keep the python MeshcatVisualizer around, because plotly RRT
    visualizer does not work with drake's CPP-based MeshcatVisualizer.
    """

    NoVis = 0
    Cpp = 1
    Python = 2


class QuasistaticSimulator:
    def __init__(
        self,
        model_directive_path: str,
        robot_stiffness_dict: Dict[str, np.ndarray],
        object_sdf_paths: Dict[str, str],
        sim_params: QuasistaticSimParameters,
        internal_vis: InternalVisualizationType = InternalVisualizationType.NoVis,
    ):
        """
        Assumptions:
        - Each rigid body has one contact geometry.
        :param robot_stiffness_dict: key: model instance name; value: 1D
            array of the stiffness of each joint in the model.
        :param object_sdf_paths: key: object model instance name; value:
            object sdf path.
        :param internal_vis: if true, a python-based MeshcatVisualizer is
            added to self.diagram.
        """
        self.sim_params = sim_params
        # Construct diagram system for proximity queries, Jacobians.
        builder = DiagramBuilder()
        (
            plant,
            scene_graph,
            robot_models,
            object_models,
        ) = self.create_plant_with_robots_and_objects(
            builder=builder,
            model_directive_path=model_directive_path,
            robot_names=[name for name in robot_stiffness_dict.keys()],
            object_sdf_paths=object_sdf_paths,
            time_step=1e-3,  # Only useful for MBP simulations.
            gravity=sim_params.gravity,
        )

        # visualization.
        self.internal_vis = internal_vis
        if internal_vis == InternalVisualizationType.Cpp:
            self.meshcat = StartMeshcat()
            self.viz = MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat
            )
            # ContactVisualizer
            self.contact_viz = ContactVisualizer.AddToBuilder(
                builder, plant, self.meshcat
            )
        elif internal_vis == InternalVisualizationType.Python:
            self.viz = ConnectMeshcatVisualizerPy(builder, scene_graph)
            self.viz.load()
            self.contact_viz = None

        diagram = builder.Build()
        self.diagram = diagram
        self.plant = plant
        self.scene_graph = scene_graph
        self.inspector = scene_graph.model_inspector()

        self.context = diagram.CreateDefaultContext()
        self.context_plant = diagram.GetMutableSubsystemContext(
            plant, self.context
        )
        self.context_sg = diagram.GetMutableSubsystemContext(
            scene_graph, self.context
        )

        # Internal visualization is used when QuasistaticSimulator is used
        # outside the Systems framework.
        if internal_vis:
            self.context_meshcat = diagram.GetMutableSubsystemContext(
                self.viz, self.context
            )
            if self.contact_viz is not None:
                self.context_meshcat_contact = (
                    diagram.GetMutableSubsystemContext(
                        self.contact_viz, self.context
                    )
                )

        self.models_unactuated = object_models
        self.models_actuated = robot_models
        self.models_all = object_models.union(robot_models)

        # body indices for each model in self.models_list.
        self.body_indices = dict()
        # velocity indices (into the generalized velocity vector of the MBP)
        self.velocity_indices = dict()
        self.position_indices = dict()
        self.n_v_dict = dict()
        self.n_q_dict = dict()
        self.n_v = plant.num_velocities()
        self.n_q = plant.num_positions()

        n_v = 0
        for model in self.models_all:
            velocity_indices = self.get_velocity_indices_for_model(model)
            position_indices = self.get_position_indices_for_model(model)
            self.velocity_indices[model] = velocity_indices
            self.position_indices[model] = position_indices
            self.n_v_dict[model] = len(velocity_indices)
            self.n_q_dict[model] = len(position_indices)
            self.body_indices[model] = plant.GetBodyIndices(model)

            n_v += len(velocity_indices)

        self.nd_per_contact = sim_params.nd_per_contact
        # Sanity check.
        assert plant.num_velocities() == n_v

        # stiffness matrices.
        self.K_a = dict()
        min_K_a_list = []
        for i, model in enumerate(self.models_actuated):
            model_name = plant.GetModelInstanceName(model)
            joint_stiffness = robot_stiffness_dict[model_name]
            assert self.n_v_dict[model] == joint_stiffness.size
            self.K_a[model] = np.diag(joint_stiffness).astype(float)
            min_K_a_list.append(np.min(joint_stiffness))
        self.min_K_a = np.min(min_K_a_list)

        # Find planar model instances.
        # TODO: it is assumed that each un-actuated model instance contains
        #  only one rigid body.
        self.is_3d_floating = dict()
        for model in self.models_unactuated:
            n_v = self.n_v_dict[model]
            n_q = plant.num_positions(model)

            if n_v == 6 and n_q == 7:
                body_indices = self.plant.GetBodyIndices(model)
                assert len(body_indices) == 1
                assert self.plant.get_body(body_indices[0]).is_floating()
                self.is_3d_floating[model] = True
            else:
                self.is_3d_floating[model] = False

        for model in self.models_actuated:
            self.is_3d_floating[model] = False

        # solver
        if is_mosek_gurobi_available():
            self.solver_qp = GurobiSolver()
            self.solver_cone = MosekSolver()
        else:
            self.solver_qp = OsqpSolver()
            self.solver_cone = ScsSolver()

        self.solver_qp_log = QpLogBarrierSolver()
        self.options_grb = SolverOptions()
        self.options_grb.SetOption(GurobiSolver.id(), "QCPDual", 1)

        # step function dictionary
        self.step_function_dict = {
            ForwardDynamicsMode.kQpMp: self.step_qp_mp,
            ForwardDynamicsMode.kQpCvx: self.step_qp_cvx,
            ForwardDynamicsMode.kLogPyramidCvx: self.step_log_cvx,
            ForwardDynamicsMode.kLogPyramidMp: self.step_log_mp,
        }

        """
        Both self.contact_results and self.query_object are updated by calling
        self.step(...)
        """
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
        # gradient from active constraints.
        self.dqp_active = QpDerivativesKktActive()

    @staticmethod
    def copy_sim_params(params_from: QuasistaticSimParameters):
        return copy.deepcopy(params_from)

    @staticmethod
    def check_params_validity(q_params: QuasistaticSimParameters):
        gm = q_params.gradient_mode
        if q_params.nd_per_contact > 2 and gm == GradientMode.kAB:
            raise RuntimeError(
                "Computing A matrix for 3D systems is not yet " "supported."
            )

        if q_params.unactuated_mass_scale == 0:
            if gm == GradientMode.kAB or gm == GradientMode.kBOnly:
                raise RuntimeError(
                    "Dynamics gradient cannot be computed when "
                    "the object has infinite mass."
                )

        if q_params.unactuated_mass_scale == np.inf:
            raise RuntimeError(
                "Setting mass matrix to 0 should be achieved "
                "using the is_quasi_dynamic flag."
            )

        log_forward_modes = {
            ForwardDynamicsMode.kLogPyramidCvx,
            ForwardDynamicsMode.kLogPyramidMp,
            ForwardDynamicsMode.kLogIcecream,
        }

        if q_params.forward_mode in log_forward_modes:
            if np.isnan(q_params.log_barrier_weight):
                raise RuntimeError(
                    f"Log barrier weight is nan when running in"
                    f" {q_params.forward_mode}."
                )

            if gm == GradientMode.kAB:
                raise RuntimeError(
                    f"Computing A for mode {gm} is not supported "
                )

    def get_sim_parmas_copy(self):
        return self.copy_sim_params(self.sim_params)

    def get_plant(self):
        return self.plant

    def get_scene_graph(self):
        return self.scene_graph

    def get_all_models(self):
        return self.models_all

    def get_actuated_models(self):
        return self.models_actuated

    def get_Dq_nextDq(self):
        return np.array(self.Dq_nextDq)

    def get_Dq_nextDqa_cmd(self):
        return np.array(self.Dq_nextDqa_cmd)

    def get_positions(self, model: ModelInstanceIndex):
        return self.plant.GetPositions(self.context_plant, model)

    def get_position_indices(self):
        return self.position_indices

    def get_query_object(self):
        return self.query_object

    def get_contact_results(self):
        return self.contact_results

    def get_velocity_indices(self):
        return self.velocity_indices

    def num_actuated_dofs(self):
        return np.sum([self.n_v_dict[model] for model in self.models_actuated])

    def num_unactuated_dof(self):
        return np.sum(
            [self.n_v_dict[model] for model in self.models_unactuated]
        )

    def get_dynamics_derivatives(self):
        return (
            np.copy(self.Dv_nextDb),
            np.copy(self.Dv_nextDe),
            np.copy(self.Dq_nextDq),
            np.copy(self.Dq_nextDqa_cmd),
        )

    def get_model_instance_name_to_index_map(self):
        name_to_model = dict()
        for model in self.models_all:
            name_to_model[self.plant.GetModelInstanceName(model)] = model
        return name_to_model

    def update_mbp_positions_from_vector(self, q: np.ndarray):
        self.plant.SetPositions(self.context_plant, q)
        # Update query object.
        self.query_object = self.scene_graph.get_query_output_port().Eval(
            self.context_sg
        )

    def update_mbp_positions(
        self, q_dict: Dict[ModelInstanceIndex, np.ndarray]
    ):
        """
        :param q_dict: A dict of np arrays keyed by model instance indices.
            Each array is the configuration of a model instance in
            self.models_list.
        :return:
        """
        # Update state in plant_context
        assert len(q_dict) == len(self.models_all)

        for model_instance_idx, q in q_dict.items():
            self.plant.SetPositions(self.context_plant, model_instance_idx, q)

        # Update query object.
        self.query_object = self.scene_graph.get_query_output_port().Eval(
            self.context_sg
        )

    def get_mbp_positions(self):
        """
        :return: a dictionary containing the current positions of all model
            instances stored in self.context_plant, keyed by
            ModelInstanceIndex.
        """
        return {
            model: self.plant.GetPositions(self.context_plant, model)
            for model in self.models_all
        }

    def get_mbp_positions_as_vec(self):
        return self.plant.GetPositions(self.context_plant)

    def draw_current_configuration(self, draw_forces=False):
        if self.internal_vis == InternalVisualizationType.NoVis:
            raise RuntimeWarning(
                "QuasistaticSimulator cannot draw because it does "
                "not own a MeshcatVisualizer."
            )
        elif self.internal_vis == InternalVisualizationType.Python:
            self.viz.DoPublish(self.context_meshcat, [])
        else:
            # CPP meshcat.
            # Body poses
            self.viz.ForcedPublish(self.context_meshcat)

            # Contact forces
            if draw_forces:
                self.contact_viz.GetInputPort("contact_results").FixValue(
                    self.context_meshcat_contact,
                    AbstractValue.Make(self.contact_results),
                )
                self.contact_viz.ForcedPublish(self.context_meshcat_contact)

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
        tau_ext_a_dict = (
            self.get_generalized_force_from_external_spatial_force(easf_list)
        )
        return {**tau_ext_a_dict, **tau_ext_u_dict}

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
        context: Context = None,
    ):
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
            frame_E=plant.world_frame(),
        )
        if position_indices is not None:
            J_WBi = J_WBi[:, position_indices]

        Jn[i_c] += n_W.dot(J_WBi)
        Jf[i_f_start : i_f_start + n_di] += d_W.dot(J_WBi)

    def find_model_instance_index_for_body(self, body):
        for model, body_indices in self.body_indices.items():
            if body.index() in body_indices:
                return model

    def calc_gravity_for_unactuated_models(self):
        gravity_all = self.plant.CalcGravityGeneralizedForces(
            self.context_plant
        )
        return {
            model: gravity_all[self.velocity_indices[model]]
            for model in self.models_unactuated
        }

    def get_generalized_force_from_external_spatial_force(
        self, easf_list: List[ExternallyAppliedSpatialForce]
    ):
        # TODO: test this more thoroughly.
        tau_ext_actuated = {
            model: np.zeros(self.n_v_dict[model])
            for model in self.models_actuated
        }

        for easf in easf_list:
            body = self.plant.get_body(easf.body_index)
            model = body.model_instance()
            assert model in self.models_actuated

            F_Bq_W = easf.F_Bq_W.get_coeffs()  # [tau, force]
            J = self.plant.CalcJacobianSpatialVelocity(
                context=self.context_plant,
                with_respect_to=JacobianWrtVariable.kV,
                frame_B=body.body_frame(),
                p_BoBp_B=easf.p_BoBq_B,
                frame_A=self.plant.world_frame(),
                frame_E=self.plant.world_frame(),
            )

            tau_ext_actuated[model] += J[
                :, self.velocity_indices[model]
            ].T.dot(F_Bq_W)

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
        signed_distance_pairs = (
            query_object.ComputeSignedDistancePairwiseClosestPoints(
                contact_detection_tolerance
            )
        )

        n_c = len(signed_distance_pairs)
        n_d = np.full(n_c, self.nd_per_contact)
        n_f = n_d.sum()
        U = np.zeros(n_c)

        phi = np.zeros(n_c)
        Jn = np.zeros((n_c, self.n_v))
        Jf = np.zeros((n_f, self.n_v))

        contact_info_list = []

        i_f_start = 0
        for i_c, sdp in enumerate(signed_distance_pairs):
            """
            A and B denote the body frames of bodyA and bodyB.
            Fa/b is the contact geometry frame relative to the body to which
                the contact geometry belongs. sdp.p_ACa is relative to frame
                Fa (geometry frame), not frame A (body frame).
            p_ACa_A is the coordinates of the "contact" point Ca relative
                to the body frame A expressed in frame A.
            """

            phi[i_c] = sdp.distance
            U[i_c] = self.get_friction_coefficient_for_signed_distance_pair(
                sdp
            )
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
                """
                When a contact pair exists between an unactuated body and
                 an actuated body, we need dC_a_W = -dC_u_W. In contrast,
                 if a contact pair contains a body that is neither actuated nor
                 unactuated, e.g. the ground, we do not need dC_a_W = -dC_u_W.

                As CalcTangentVectors(n, nd) != - CalcTangentVectors(-n, nd),
                    care is needed to ensure that the conditions above are met.

                The normal n_A/B_W needs to point into body A/B, respectively.
                """
                n_A_W = sdp.nhat_BA_W
                d_A_W = calc_tangent_vectors(n_A_W, n_d[i_c])
                n_B_W = -n_A_W
                d_B_W = -d_A_W

                # new Jn and Jf
                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyA,
                    pC_D=p_ACa_A,
                    n_W=n_A_W,
                    d_W=d_A_W,
                    i_c=i_c,
                    n_di=n_d[i_c],
                    i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn,
                    Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV,
                )

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyB,
                    pC_D=p_BCb_B,
                    n_W=n_B_W,
                    d_W=d_B_W,
                    i_c=i_c,
                    n_di=n_d[i_c],
                    i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn,
                    Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV,
                )

            elif is_B_in:
                n_B_W = -sdp.nhat_BA_W
                d_B_W = calc_tangent_vectors(n_B_W, n_d[i_c])

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyB,
                    pC_D=p_BCb_B,
                    n_W=n_B_W,
                    d_W=d_B_W,
                    i_c=i_c,
                    n_di=n_d[i_c],
                    i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn,
                    Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV,
                )

            elif is_A_in:
                n_A_W = sdp.nhat_BA_W
                d_A_W = calc_tangent_vectors(n_A_W, n_d[i_c])

                self.update_normal_and_tangential_jacobian_rows(
                    body=bodyA,
                    pC_D=p_ACa_A,
                    n_W=n_A_W,
                    d_W=d_A_W,
                    i_c=i_c,
                    n_di=n_d[i_c],
                    i_f_start=i_f_start,
                    position_indices=None,
                    Jn=Jn,
                    Jf=Jf,
                    jacobian_wrt_variable=JacobianWrtVariable.kV,
                )
            else:
                # either A or B is in self.body_indices_list
                raise RuntimeError(
                    "At least one body in a contact pair "
                    "should be in self.body_indices_list."
                )

            i_f_start += n_d[i_c]

            # Store contact positions in order to draw contact forces later.
            # TODO: contact forces at step (l+1) is drawn with the
            # configuration at step l.
            if is_A_in:
                X_WD = self.plant.EvalBodyPoseInWorld(
                    self.context_plant, bodyA
                )
                contact_info_list.append(
                    MyContactInfo(
                        bodyA_index=bodyB.index(),
                        bodyB_index=bodyA.index(),
                        geometry_id_A=sdp.id_B,
                        geometry_id_B=sdp.id_A,
                        p_WC_W=X_WD.multiply(p_ACa_A),
                        n_W=n_A_W,
                        dC_W=d_A_W,
                    )
                )
            elif is_B_in:
                X_WD = self.plant.EvalBodyPoseInWorld(
                    self.context_plant, bodyB
                )
                contact_info_list.append(
                    MyContactInfo(
                        bodyA_index=bodyA.index(),
                        bodyB_index=bodyB.index(),
                        geometry_id_A=sdp.id_A,
                        geometry_id_B=sdp.id_B,
                        p_WC_W=X_WD.multiply(p_BCb_B),
                        n_W=n_B_W,
                        dC_W=d_B_W,
                    )
                )
            else:
                raise RuntimeError(
                    "At least one body in a contact pair "
                    "should be unactuated."
                )

        return n_c, n_d, n_f, Jn, Jf, phi, U, contact_info_list

    def update_contact_results(
        self,
        my_contact_info_list: List[MyContactInfo],
        beta: np.ndarray,
        h: float,
        n_c: int,
        n_d: np.ndarray,
        mu_list: np.ndarray,
    ):
        assert len(my_contact_info_list) == n_c
        contact_results = ContactResults()
        i_f_start = 0
        for i_c, my_contact_info in enumerate(my_contact_info_list):
            i_f_end = i_f_start + n_d[i_c]
            beta_i = beta[i_f_start:i_f_end]
            f_normal_W = my_contact_info.n_W * beta_i.sum() / h
            f_tangential_W = (
                my_contact_info.dC_W.T.dot(beta_i) * mu_list[i_c] / h
            )
            point_pair = PenetrationAsPointPair()
            point_pair.id_A = my_contact_info.geometry_id_A
            point_pair.id_B = my_contact_info.geometry_id_B
            contact_results.AddContactInfo(
                PointPairContactInfo(
                    my_contact_info.bodyA_index,
                    my_contact_info.bodyB_index,
                    f_normal_W + f_tangential_W,
                    my_contact_info.p_WC_W,
                    0,
                    0,
                    point_pair,
                )
            )

            i_f_start += n_d[i_c]

        self.contact_results = contact_results

    def get_mbp_body_from_scene_graph_geometry(self, g_id):
        f_id = self.inspector.GetFrameId(g_id)
        return self.plant.GetBodyFromFrameId(f_id)

    def get_position_indices_for_model(self, model_instance_index):
        selector = np.arange(self.plant.num_positions())
        return self.plant.GetPositionsFromArray(
            model_instance_index, selector
        ).astype(int)

    def get_velocity_indices_for_model(self, model_instance_index):
        selector = np.arange(self.plant.num_velocities())
        return self.plant.GetVelocitiesFromArray(
            model_instance_index, selector
        ).astype(int)

    def get_friction_coefficient_for_signed_distance_pair(self, sdp):
        props_A = self.inspector.GetProximityProperties(sdp.id_A)
        props_B = self.inspector.GetProximityProperties(sdp.id_B)
        cf_A = props_A.GetProperty("material", "coulomb_friction")
        cf_B = props_B.GetProperty("material", "coulomb_friction")
        cf = CalcContactFrictionFromSurfaceProperties(cf_A, cf_B)
        return cf.static_friction()

    def calc_jacobian_and_phi(self, contact_detection_tolerance):
        (
            n_c,
            n_d,
            n_f,
            Jn,
            Jf,
            phi_l,
            U,
            contact_info_list,
        ) = self.calc_contact_jacobians(contact_detection_tolerance)

        phi_constraints = np.zeros(n_f)
        J = np.zeros_like(Jf)
        j_start = 0
        for i_c in range(n_c):
            for j in range(n_d[i_c]):
                idx = j_start + j
                J[idx] = Jn[i_c] + U[i_c] * Jf[idx]
                phi_constraints[idx] = phi_l[i_c]
            j_start += n_d[i_c]

        return (
            phi_constraints,
            J,
            phi_l,
            Jn,
            contact_info_list,
            n_c,
            n_d,
            n_f,
            U,
        )

    @staticmethod
    def check_cvx_status(status: str):
        if status != "optimal":
            if status == "optimal_inaccurate":
                warnings.warn("CVX solver is inaccurate.")
            else:
                raise RuntimeError("CVX solver status is {}".format(status))

    @staticmethod
    def set_sim_params(params: QuasistaticSimParameters, **kwargs):
        for name, value in kwargs.items():
            params.__setattr__(name, value)

    @staticmethod
    def convert_sim_params_into_dict(params: QuasistaticSimParameters):
        q_sim_params_dict = {
            name: params.__getattribute__(name)
            for name in params.__dir__()
            if not name.startswith("_")
        }
        return q_sim_params_dict

    def calc_scaled_mass_matrix(self, h: float, unactuated_mass_scale: float):
        """
        "Mass" in quasi-dynamic dynamics should not be interpreted as
        inertia that affects acceleration from a given force, as velocity
        does not exist in the quasi-dynamic world.

        Instead, it makes more sense to interpret mass as a regularization
        which keep un-actuated objects still when there is no force acting on
        them, which is consistent with Newton's 1st Law.

        However, having a mass matrix slows down the motion of un-actuated
        objects when it interacts with actuated objects. Therefore, the mass
        matrix should be as small as possible without causing numerical issues.

        The strategy used here is to scale the true mass matrix of
         un-actuated objects by epsilon, so that the largest eigen value of the
         mass matrix is a constant (unactuated_mass_scale) times smaller than
         the smallest eigen value of (h^2 * K), where h is the simulation time
         step and K the stiffness matrix of the robots.

        With this formulation, the Q matrix in the QP is given by
        [[epsilon * M_u, 0    ],
         [0            , h^2 K]],
        where
        max_M_u_eigen_value * epsilon * unactuated_mass_scale == min_h_squared_K

        Special case I, unactuated_mass_scale == 0 means that the object
         position remains fixed, as if it has infinite inertia. This is
         useful for grasp sampling, when we want the object pose to
         remain fixed while moving the fingers to make contact. But having
         infs in the QP is bad. Instead, the original M_u is used,
         and the infinite inertia effect is achieved by not updating q_u in
         self.step_configuration(...).

        Special case II, unactuated_mass_scale == INFINITY, if interpreted
         literally, would set M_u to 0, thus having the same effect as
         setting is_quasi_dynamic to false. We don't want to inadvertently
         disable mass, so an exception is thrown in this case.

        Special case III, unactuated_mass_scale == NAN. The original mass
         matrix is used, without any scaling.
        """
        M = self.plant.CalcMassMatrix(self.context_plant)
        M_u_dict = {}
        for model in self.models_unactuated:
            idx_v_model = self.velocity_indices[model]
            M_u_dict[model] = M[idx_v_model[:, None], idx_v_model]

        if unactuated_mass_scale == 0 or np.isnan(unactuated_mass_scale):
            return M_u_dict

        max_eigen_value_M_u = {
            model: np.max(M_u.diagonal()) for model, M_u in M_u_dict.items()
        }

        min_K_a_h2 = self.min_K_a * h**2

        for model, M_u in M_u_dict.items():
            epsilon = (
                min_K_a_h2 / max_eigen_value_M_u[model] / unactuated_mass_scale
            )
            M_u *= epsilon

        return M_u_dict

    def form_Q_and_tau_h(
        self,
        q_dict: Dict[ModelInstanceIndex, np.ndarray],
        q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
        tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
        h: float,
        unactuated_mass_scale: float,
    ):
        Q = np.zeros((self.n_v, self.n_v))
        tau_h = np.zeros(self.n_v)

        # TODO: make this an input to the function, so that there exists a
        #  version of self.step which is independent of self.sim_params.
        if self.sim_params.is_quasi_dynamic:
            M_u_dict = self.calc_scaled_mass_matrix(h, unactuated_mass_scale)

        for model in self.models_unactuated:
            idx_v_model = self.velocity_indices[model]
            tau_h[idx_v_model] = tau_ext_dict[model] * h

            if self.sim_params.is_quasi_dynamic:
                Q[idx_v_model[:, None], idx_v_model] = M_u_dict[model]

        idx_i, idx_j = np.diag_indices(self.n_v)
        for model in self.models_actuated:
            idx_v_model = self.velocity_indices[model]
            dq_a_cmd = q_a_cmd_dict[model] - q_dict[model]
            tau_a = self.K_a[model].dot(dq_a_cmd) + tau_ext_dict[model]
            tau_h[idx_v_model] = tau_a * h

            Q[idx_i[idx_v_model], idx_j[idx_v_model]] = (
                self.K_a[model].diagonal() * h**2
            )
        return Q, tau_h

    def step_qp_mp(
        self,
        h: float,
        phi_constraints: np.ndarray,
        J: np.ndarray,
        Q: np.ndarray,
        tau_h: np.ndarray,
        gradient_mode: GradientMode,
        **kwargs,
    ):
        prog = MathematicalProgram()
        # generalized velocity times time step.
        v = prog.NewContinuousVariables(self.n_v, "v")

        prog.AddQuadraticCost(Q, -tau_h, v)
        e = phi_constraints / h
        constraints = prog.AddLinearConstraint(
            A=-J, lb=np.full_like(phi_constraints, -np.inf), ub=e, vars=v
        )

        result = self.solver_qp.Solve(prog, None, self.options_grb)
        # self.optimizer_time_log.append(
        #     result.get_solver_details().optimizer_time)
        assert result.is_success()
        beta = np.zeros(0)
        if J.shape[0] > 0:
            beta = -result.GetDualSolution(constraints)
            beta = np.array(beta).squeeze()

        # extract v_h from vector into a dictionary.
        v_values = result.GetSolution(v)
        v_h_value_dict = dict()
        for model in self.models_all:
            indices = self.velocity_indices[model]
            v_h_value_dict[model] = v_values[indices] * h

        # Gradient
        if gradient_mode == GradientMode.kNone:
            DvDb, DvDe = None, None
        else:
            # lambda_threshold: impulse generated by a force of 0.1N during
            #  h.
            dqp = self.dqp_active
            dqp.update_problem(
                Q=Q,
                b=-tau_h,
                G=-J,
                e=e,
                z_star=v_values,
                lambda_star=beta,
                lambda_threshold=0.1 * h,
            )

            if gradient_mode == GradientMode.kAB:
                DvDb = dqp.calc_DzDb()
                DvDe = dqp.calc_DzDe()
            elif gradient_mode == GradientMode.kBOnly:
                DvDb = dqp.calc_DzDb()
                DvDe = None
            else:
                raise RuntimeError()

        return v_h_value_dict, beta, DvDb, DvDe

    def step_log_mp(
        self,
        h: float,
        phi_constraints: np.ndarray,
        J: np.ndarray,
        Q: np.ndarray,
        tau_h: np.ndarray,
        gradient_mode: GradientMode,
        log_barrier_weight: float,
    ):
        m = len(phi_constraints)
        n_v = self.n_v

        prog = MathematicalProgram()
        v = prog.NewContinuousVariables(n_v, "v")
        s = prog.NewContinuousVariables(m, "s")

        # Costs.
        prog.AddQuadraticCost(Q=Q, b=-tau_h, vars=v, is_convex=True)
        prog.AddLinearCost(a=-np.full(m, 1 / log_barrier_weight), b=0, vars=s)

        # exponential cone constraints for contacts.
        for i in range(m):
            A = np.zeros([3, n_v + 1])
            A[0, :n_v] = J[i]
            A[2, -1] = 1

            b = np.array([phi_constraints[i] / h, 1, 0])
            prog.AddExponentialConeConstraint(
                A=A, b=b, vars=np.hstack([v, [s[i]]])
            )

        result = self.solver_cone.Solve(prog, None, None)
        assert result.is_success()

        # extract v_h from vector into a dictionary.
        v_values = result.GetSolution(v)
        v_h_value_dict = dict()
        for model in self.models_all:
            indices = self.velocity_indices[model]
            v_h_value_dict[model] = v_values[indices] * h

        # no dual variables yet.
        beta = np.zeros(m)

        # no gradients yet
        DvDb, DvDe = None, None

        return v_h_value_dict, beta, DvDb, DvDe

    def step_qp_cvx(
        self,
        h: float,
        phi_constraints: np.ndarray,
        J: np.ndarray,
        Q: np.ndarray,
        tau_h: np.ndarray,
        gradient_mode: GradientMode,
        **kwargs,
    ):
        # Make a CVX problem.
        # The cholesky decomposition is needed because cp.sum_squares() is the
        # only way I've found so far to ensure the problem is DCP (
        # disciplined convex program).
        """
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
        """
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
            constraints,
        )

        prob.solve(
            requires_grad=gradient_mode != GradientMode.kNone, solver="GUROBI"
        )
        self.check_cvx_status(prob.status)

        # extract v_h from vector into a dictionary.
        v_h_value_dict = dict()
        for model in self.models_all:
            indices = self.velocity_indices[model]
            v_h_value_dict[model] = v.value[indices] * h

        # Gradient.
        if gradient_mode == GradientMode.kNone:
            DvDb, DvDe = None, None
        else:
            DvDb = np.zeros((self.n_v, self.n_v))
            DvDe = np.zeros((self.n_v, n_e))
            for i in range(self.n_v):
                dv = np.zeros(self.n_v)
                dv[i] = 1
                v.gradient = dv
                prob.backward()

                DvDb[i] = b_cp.gradient
                DvDe[i] = e_cp.gradient

            if gradient_mode == GradientMode.kBOnly:
                DvDe = None
            elif gradient_mode == GradientMode.kAB:
                pass
            else:
                raise RuntimeError()

        return v_h_value_dict, np.zeros_like(phi_constraints), DvDb, DvDe

    def step_log_cvx(
        self,
        h: float,
        phi_constraints: np.ndarray,
        J: np.ndarray,
        Q: np.ndarray,
        tau_h: np.ndarray,
        gradient_mode: GradientMode,
        log_barrier_weight,
    ):
        v = cp.Variable(self.n_v)

        log_barriers_sum = 0.0
        if len(phi_constraints) > 0:
            log_barriers_sum = cp.sum(cp.log(phi_constraints / h + J @ v))
        prob = cp.Problem(
            cp.Minimize(
                0.5 * cp.quad_form(v, Q)
                - tau_h @ v
                - log_barriers_sum / log_barrier_weight
            )
        )

        prob.solve()
        self.check_cvx_status(prob.status)

        # extract v_h from vector into a dictionary.
        v_h_value_dict = dict()
        for model in self.models_all:
            indices = self.velocity_indices[model]
            v_h_value_dict[model] = v.value[indices] * h

        # TODO: gradient not supported yet.
        DvDb, DvDe = None, None
        return v_h_value_dict, np.zeros_like(phi_constraints), DvDb, DvDe

    def step(
        self,
        q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
        tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
        sim_params: QuasistaticSimParameters,
    ):
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
        :param forward_mode: one of {'qp_mp', 'qp_cvx', 'unconstrained'}.
        :param gradient_mode: whether gradient of the dynamics is computed.
        :param grad_from_active_constraints: whether gradient is computed only
           from active constraints.
        :return: system configuration at the next time step, stored in a
            dictionary keyed by ModelInstanceIndex.
        """
        # Unpack some parameters.
        h = sim_params.h
        unactuated_mass_scale = sim_params.unactuated_mass_scale
        forward_mode = sim_params.forward_mode
        gradient_mode = sim_params.gradient_mode

        # Forward dynamics.
        q_dict = self.get_mbp_positions()

        Q, tau_h = self.form_Q_and_tau_h(
            q_dict,
            q_a_cmd_dict,
            tau_ext_dict,
            h,
            unactuated_mass_scale=unactuated_mass_scale,
        )

        (
            phi_constraints,
            J,
            phi_l,
            Jn,
            contact_info_list,
            n_c,
            n_d,
            n_f,
            U,
        ) = self.calc_jacobian_and_phi(
            self.sim_params.contact_detection_tolerance
        )

        v_h_value_dict, beta, Dv_nextDb, Dv_nextDe = self.step_function_dict[
            forward_mode
        ](
            h,
            phi_constraints,
            J,
            Q,
            tau_h,
            gradient_mode,
            log_barrier_weight=sim_params.log_barrier_weight,
        )

        dq_dict = dict()
        for model in self.models_actuated:
            v_h_value = v_h_value_dict[model]
            dq_dict[model] = v_h_value

        for model in self.models_unactuated:
            v_h_value = v_h_value_dict[model]
            if self.is_3d_floating[model]:
                q_u = q_dict[model]
                Q_WB = q_u[:4]  # Quaternion Q_WB

                dq_u = np.zeros(7)
                dq_u[:4] = self.get_E(Q_WB).dot(v_h_value[:3])
                dq_u[4:] = v_h_value[3:]
                dq_dict[model] = dq_u
            else:
                dq_dict[model] = v_h_value

        # Normalize quaternions and update simulator context.
        # TODO: normalization should happen after computing the derivatives.
        #  But doing it the other way around seems to lead to smaller
        #  difference between numerical and analytic derivatives. Find out why.
        self.step_configuration(q_dict, dq_dict, unactuated_mass_scale)
        self.update_mbp_positions(q_dict)
        if hasattr(ContactResults, "AddContactInfo"):
            self.update_contact_results(
                contact_info_list, beta, h, n_c, n_d, U
            )

        # Gradients.
        """
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
        """
        self.Dv_nextDb = Dv_nextDb
        self.Dv_nextDe = Dv_nextDe

        if (
            forward_mode == ForwardDynamicsMode.kQpMp
            or forward_mode == ForwardDynamicsMode.kQpCvx
        ):
            self.backward_qp(
                gradient_mode=gradient_mode,
                h=h,
                q_dict=q_dict,
                n_f=n_f,
                n_c=n_c,
                n_d=n_d,
                Jn=Jn,
            )

        elif (
            forward_mode == ForwardDynamicsMode.kLogPyramidMp
            or forward_mode == ForwardDynamicsMode.kLogPyramidCvx
        ):
            self.backward_log(
                gradient_mode=gradient_mode,
                h=h,
                v_h_dict=v_h_value_dict,
                Q=Q,
                J=J,
                phi_constraints=phi_constraints,
                log_barrier_weight=self.sim_params.log_barrier_weight,
            )
        else:
            raise NotImplementedError(
                f"{self.sim_params.mode} is not supported."
            )

        return q_dict

    def backward_qp(
        self,
        gradient_mode: GradientMode,
        h: float,
        q_dict: Dict[ModelInstanceIndex, np.ndarray],
        n_f: int,
        n_c: int,
        n_d: int,
        Jn: np.ndarray,
    ):
        if gradient_mode == GradientMode.kNone:
            return

        Dv_nextDb = self.Dv_nextDb
        Dv_nextDe = self.Dv_nextDe
        if gradient_mode == GradientMode.kBOnly:
            self.Dq_nextDqa_cmd = self.calc_dfdu(Dv_nextDb, h, q_dict)
            self.Dq_nextDq = np.zeros([self.n_v, self.n_v])
        elif gradient_mode == GradientMode.kAB:
            self.Dq_nextDqa_cmd = self.calc_dfdu(Dv_nextDb, h, q_dict)
            self.Dq_nextDq = self.calc_dfdx(
                Dv_nextDb=Dv_nextDb,
                Dv_nextDe=Dv_nextDe,
                h=h,
                n_f=n_f,
                n_c=n_c,
                n_d=n_d,
                Jn=Jn,
            )

    def backward_log(
        self,
        gradient_mode: GradientMode,
        h: float,
        v_h_dict: Dict[ModelInstanceIndex, np.ndarray],
        Q: np.ndarray,
        J: np.ndarray,
        phi_constraints: np.ndarray,
        log_barrier_weight: float,
    ):
        if gradient_mode == GradientMode.kNone:
            return

        if gradient_mode == GradientMode.kBOnly:
            # recover vstar from vstar_dict...
            vstar = np.zeros(self.n_v)
            for model, v_h_model in v_h_dict.items():
                vstar[self.velocity_indices[model]] = v_h_model / h

            # Use implicit function theorem to get DvDu.
            dydu = np.zeros_like(Q)
            for j in range(J.shape[1]):
                for i in range(J.shape[0]):
                    dydu[j] += (
                        J[i]
                        * J[i, j]
                        / (phi_constraints[i] / h + J[i, :] @ vstar) ** 2.0
                    )

            coeff = Q + dydu / log_barrier_weight
            bias = np.zeros((self.n_v, self.n_v))
            idx_i, idx_j = np.diag_indices(self.n_v)
            idx_u = []
            for model in self.models_actuated:
                idx_v_model = self.velocity_indices[model]
                idx_u += list(idx_v_model)
                bias[idx_i[idx_v_model], idx_j[idx_v_model]] = (
                    self.K_a[model].diagonal() * h
                )

            DvDu = h * np.linalg.solve(coeff, bias)[:, idx_u]

            self.Dq_nextDqa_cmd = DvDu
            self.Dq_nextDq = np.zeros([self.n_v, self.n_v])

            return

        # gradient_mode == GradientMode.kAB:
        raise NotImplementedError(
            "GradientMode.kAB is not implemented for log barrier dynamics."
        )

    def step_default(
        self,
        q_a_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
        tau_ext_dict: Dict[ModelInstanceIndex, np.ndarray],
    ):
        """
        Steps the dynamics forward by h using the params in self.sim_params.
        """
        return self.step(
            q_a_cmd_dict=q_a_cmd_dict,
            tau_ext_dict=tau_ext_dict,
            sim_params=self.sim_params,
        )

    @staticmethod
    def get_E(Q_AB: np.ndarray):
        """
        Let w_AB_A denote the angular velocity of frame B relative to frame A
         expressed in A, and Q_AB the unit quaternion representing the
         orientation of frame B relative to frame A. The time derivative of
         Q_AB, D(Q_AB)Dt, can be written as a linear function of w_AB_A:
            D(Q_AB)Dt = E @ w_AB_A.
        This function computes E from Q_AB.

        Reference: Appendix A.3 of Natale's book
            "Interaction Control of Robot Manipulators".
        """
        E = np.zeros((4, 3))
        E[0] = [-Q_AB[1], -Q_AB[2], -Q_AB[3]]
        E[1] = [Q_AB[0], Q_AB[3], -Q_AB[2]]
        E[2] = [-Q_AB[3], Q_AB[0], Q_AB[1]]
        E[3] = [Q_AB[2], -Q_AB[1], Q_AB[0]]
        E *= 0.5
        return E

    @staticmethod
    def copy_model_instance_index_dict(
        q_dict: Dict[ModelInstanceIndex, np.ndarray]
    ):
        return {key: np.array(value) for key, value in q_dict.items()}

    def calc_dfdu(
        self,
        Dv_nextDb: np.ndarray,
        h: float,
        q_dict: Dict[ModelInstanceIndex, np.ndarray],
    ):
        """
        Computes dfdu, aka the B matrix in x_next = Ax + Bu, using the chain
        rule.
        """
        n_a = self.num_actuated_dofs()
        DbDqa_cmd = np.zeros((self.n_v, n_a))

        j_start = 0
        for model in self.models_actuated:
            idx_v_model = self.velocity_indices[model]
            n_v_i = len(idx_v_model)
            idx_rows = idx_v_model[:, None]
            idx_cols = np.arange(j_start, j_start + n_v_i)[None, :]
            DbDqa_cmd[idx_rows, idx_cols] = -h * self.K_a[model]

            j_start += n_v_i

        Dv_nextDqa_cmd = Dv_nextDb @ DbDqa_cmd
        if self.n_v == self.n_q:
            return h * Dv_nextDqa_cmd

        Dq_dot_nextDqa_cmd = np.zeros((self.n_q, n_a))
        for model in self.models_all:
            idx_v_model = self.velocity_indices[model]
            idx_q_model = self.position_indices[model]
            if self.is_3d_floating[model]:
                # rotation
                Q_WB = q_dict[model][:4]
                E = self.get_E(Q_WB)
                # D = calc_normalization_derivatives(Q_WB)
                Dq_dot_nextDqa_cmd[idx_q_model[:4], :] = (
                    E @ Dv_nextDqa_cmd[idx_v_model[:3], :]
                )
                # translation
                Dq_dot_nextDqa_cmd[idx_q_model[4:], :] = Dv_nextDqa_cmd[
                    idx_v_model[3:], :
                ]
            else:
                Dq_dot_nextDqa_cmd[idx_q_model, :] = Dv_nextDqa_cmd[
                    idx_v_model, :
                ]
        return h * Dq_dot_nextDqa_cmd

    def calc_dfdx(
        self,
        Dv_nextDb: np.ndarray,
        Dv_nextDe: np.ndarray,
        h: float,
        n_f: int,
        n_c: int,
        n_d: np.ndarray,
        Jn: np.ndarray,
    ):
        """
        Computes dfdx, aka the A matrix in x_next = Ax + Bu, using the chain
            rule. Note the term that includes the partial derivatives of the
            contact Jacobian is missing in the current implementation,
            therefore the result is probably wrong when the contact normals
            are not constant.
        Nevertheless, we believe that using this term in trajectory
            optimization leads to worse convergence and won't be using it.
        """
        # TODO: for now it is assumed that n_q == n_v.
        #  Dphi_constraints_Dv is used for Dphi_constraints_Dq.
        # ---------------------------------------------------------------------
        Dphi_constraints_Dq = np.zeros((n_f, self.n_v))
        j_start = 0
        for i_c in range(n_c):
            Dphi_constraints_Dq[j_start : j_start + n_d[i_c]] = Jn[i_c]
            j_start += n_d[i_c]

        # ---------------------------------------------------------------------
        DbDq = np.zeros((self.n_v, self.n_v))
        j_start = 0
        for model in self.models_actuated:
            idx_v_model = self.velocity_indices[model]
            n_v_i = len(idx_v_model)
            idx_rows = idx_v_model[:, None]
            idx_cols = idx_v_model[None, :]
            DbDq[idx_rows, idx_cols] = h * self.K_a[model]

            j_start += n_v_i

        # Dq_nextDq
        Dv_nextDq_1 = Dv_nextDb @ DbDq
        Dv_nextDq_2 = Dv_nextDe @ Dphi_constraints_Dq / h
        Dv_nextDq = Dv_nextDq_1 + Dv_nextDq_2

        return np.eye(self.n_v) + h * Dv_nextDq

    def calc_dfdu_numerical(
        self,
        q_dict: Dict[ModelInstanceIndex, np.ndarray],
        qa_cmd_dict: Dict[ModelInstanceIndex, np.ndarray],
        du: float,
        sim_params: QuasistaticSimParameters,
    ):
        """
        Not an efficient way to compute gradients. For debugging only.
        :param q_dict: nominal state (x) of the quasistatic system.
        :param qa_cmd_dict: nominal input (u) of the quaistatic system.
        :param du: perturbation to each element of u made to compute the numerical
            gradient.
        """
        tau_ext_dict = self.calc_tau_ext([])

        n_a = self.num_actuated_dofs()
        dfdu = np.zeros((self.n_q, n_a))
        sim_params = self.copy_sim_params(sim_params)
        sim_params.gradient_mode = GradientMode.kNone

        idx_a = 0  # index for actuated DOFs (into u).
        for model_a in self.models_actuated:
            n_q_i = len(self.position_indices[model_a])

            for i in range(n_q_i):
                qa_cmd_dict_plus = self.copy_model_instance_index_dict(
                    qa_cmd_dict
                )
                qa_cmd_dict_plus[model_a][i] += du
                self.update_mbp_positions(q_dict)
                q_dict_plus = self.step(
                    q_a_cmd_dict=qa_cmd_dict_plus,
                    tau_ext_dict=tau_ext_dict,
                    sim_params=sim_params,
                )

                qa_cmd_dict_minus = self.copy_model_instance_index_dict(
                    qa_cmd_dict
                )
                qa_cmd_dict_minus[model_a][i] -= du
                self.update_mbp_positions(q_dict)
                q_dict_minus = self.step(
                    q_a_cmd_dict=qa_cmd_dict_minus,
                    tau_ext_dict=tau_ext_dict,
                    sim_params=sim_params,
                )

                for model in self.models_all:
                    idx_v_model = self.position_indices[model]
                    dfdu[idx_v_model, idx_a] = (
                        (q_dict_plus[model] - q_dict_minus[model]) / 2 / du
                    )

                idx_a += 1

        return dfdu

    def step_configuration(
        self,
        q_dict: Dict[ModelInstanceIndex, np.ndarray],
        dq_dict: Dict[ModelInstanceIndex, np.ndarray],
        unactuated_mass_scale: float,
    ):
        """
        Adds the delta of each model state, e.g. dq_u_list[i], to the
            corresponding model configuration in q_list. If q_list[i]
            includes a quaternion, the quaternion (usually the first four
            numbers of a seven-number array) is normalized.
        :return: None.
        """
        if unactuated_mass_scale > 0 or np.isnan(unactuated_mass_scale):
            for model in self.models_unactuated:
                q_u = q_dict[model]
                q_u += dq_dict[model]

                if self.is_3d_floating[model]:
                    # pass
                    q_u[:4] /= np.linalg.norm(q_u[:4])  # normalize quaternion

        for model in self.models_actuated:
            q_dict[model] += dq_dict[model]

    @staticmethod
    def create_plant_with_robots_and_objects(
        builder: DiagramBuilder,
        model_directive_path: str,
        robot_names: List[str],
        object_sdf_paths: Dict[str, str],
        time_step: float,
        gravity: np.ndarray,
    ):
        """
        Add plant and scene_graph constructed from a model_directive to builder.
        :param builder:
        :param model_directive_path:
        :param robot_names: names in this list must be consistent with the
            corresponding model directive .yml file.
        :param object_names:
        :param time_step:
        :param gravity:
        :return:
        """

        # MultibodyPlant
        plant = MultibodyPlant(time_step)
        _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
        parser = Parser(plant=plant, scene_graph=scene_graph)
        add_package_paths_local(parser)

        # Objects
        # It is important that object_models and robot_models are ordered.
        object_models = set()
        for name, sdf_path in object_sdf_paths.items():
            object_models.add(
                parser.AddModelFromFile(sdf_path, model_name=name)
            )

        # Robots
        ProcessModelDirectives(
            LoadModelDirectives(model_directive_path), plant, parser
        )
        robot_models = set()
        for name in robot_names:
            robot_model = plant.GetModelInstanceByName(name)
            robot_models.add(robot_model)

        # gravity
        plant.mutable_gravity_field().set_gravity_vector(gravity)
        plant.Finalize()

        return plant, scene_graph, robot_models, object_models
