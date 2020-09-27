from typing import List

from pydrake.common.value import AbstractValue
from pydrake.systems.meshcat_visualizer import (
    MeshcatVisualizer, MeshcatContactVisualizer)
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.multibody.plant import (PointPairContactInfo, ContactResults,
                                     CalcContactFrictionFromSurfaceProperties)
from pydrake.geometry import PenetrationAsPointPair
from pydrake.common.value import AbstractValue

from setup_environments import *
from contact_aware_control.contact_particle_filter.utils_cython import (
    CalcTangentVectors)


# %%
class MyContactInfo(object):
    """
    Used as an intermediate storage structure for constructing
    PointPairContactInfo.
    n_W is pointing into body B.
    dC_W are the tangent vectors spanning the tangent plane at the
    contact point.
    """

    def __init__(self, bodyA_index, bodyB_index, p_WC_W, n_W, dC_W):
        self.bodyA_index = bodyA_index
        self.bodyB_index = bodyB_index
        self.p_WC_W = p_WC_W
        self.n_W = n_W
        self.dC_W = dC_W


class QuasistaticSimulator:
    def __init__(self, setup_environment, nd_per_contact, object_sdf_path,
                 joint_stiffness):
        """
        Let's assume that
        - There's only one unactuated and one actuated model instance.
        - Each rigid body has one contact geometry.
        :param joint_stiffness: a 1D vector of length n_a. The stiffness of
        all joints of the robot.
        """
        self.Kq_a = np.diag(joint_stiffness).astype(float)

        # Construct diagram system for proximity queries, Jacobians.
        builder = DiagramBuilder()
        plant, scene_graph, robot_model_list, object_model_list = \
            setup_environment(builder, object_sdf_path)
        viz = MeshcatVisualizer(
            scene_graph, frames_to_draw={"three_link_arm": {"link_ee"}})
        builder.AddSystem(viz)
        builder.Connect(
            scene_graph.get_pose_bundle_output_port(),
            viz.GetInputPort("lcm_visualization"))

        # ContactVisualizer
        contact_viz = MeshcatContactVisualizer(meshcat_viz=viz, plant=plant)
        builder.AddSystem(contact_viz)
        builder.Connect(
            scene_graph.get_pose_bundle_output_port(),
            contact_viz.GetInputPort("pose_bundle"))

        diagram = builder.Build()
        viz.vis.delete()
        viz.load()

        self.diagram = diagram
        self.plant = plant
        self.scene_graph = scene_graph
        self.viz = viz
        self.contact_viz = contact_viz
        self.inspector = scene_graph.model_inspector()

        self.context = diagram.CreateDefaultContext()
        self.context_plant = diagram.GetMutableSubsystemContext(
            plant, self.context)
        self.context_sg = diagram.GetMutableSubsystemContext(
            scene_graph, self.context)
        self.context_meshcat = diagram.GetMutableSubsystemContext(
            self.viz, self.context)
        self.context_meshcat_contact = diagram.GetMutableSubsystemContext(
            self.contact_viz, self.context)

        '''
        Models list: [unactuated_model1, ...unactuated_model_n, 
            actuated_model1, actuated_modeln]. The order in this list must be 
            consistent with the order in q_list passed to other functions in 
            this class.
        '''
        # TODO: if an element of robot_model_list is a list itself,
        #  it is assumed that the models
        #  in robot_model_list are fixed into one kinematic chain. For
        #  example, a robot and a gripper fixed to the last link of the robot.
        #  It would be nice to handle this more gracefully, but I don't know
        #  how to do that yet...
        models_list = object_model_list + robot_model_list
        n_u_models = len(object_model_list)
        n_a_models = len(robot_model_list)
        self.models_unactuated_indices = np.arange(n_u_models)
        self.models_actuated_indices = n_u_models + np.arange(n_a_models)

        self.models_list_expanded = []
        for model in models_list:
            if isinstance(model, list):
                for m in model:
                    self.models_list_expanded.append(m)
            else:
                self.models_list_expanded.append(model)

        # body indices for each model in self.models_list.
        self.body_indices_list = []
        # velocity indices (into the generalized velocity vector of the MBP)
        self.velocity_indices_list = []
        n_v_list = []

        for model in models_list:
            if isinstance(model, list):
                # model is a list of model instances... yes I know it can be
                # less confusing...
                body_indices = []
                velocity_indices = []
                for m in model:
                    body_indices += plant.GetBodyIndices(m)
                    velocity_indices += \
                        self.GetVelocityIndicesForModel(m).tolist()
                self.body_indices_list.append(body_indices)
                self.velocity_indices_list.append(velocity_indices)
                n_v_list.append(len(velocity_indices))
            else:
                body_indices = plant.GetBodyIndices(model)
                velocity_indices = self.GetVelocityIndicesForModel(model)
                self.body_indices_list.append(body_indices)
                self.velocity_indices_list.append(velocity_indices)
                n_v_list.append(len(velocity_indices))

        self.n_v_list = np.array(n_v_list)
        self.models_list = models_list
        self.nd_per_contact = nd_per_contact

        # TODO: One actuated bodies for now.
        assert len(self.models_actuated_indices) == 1
        assert self.n_v_list[self.models_actuated_indices].sum() == \
               self.Kq_a.diagonal().size

        # solver
        self.solver = GurobiSolver()
        assert self.solver.available()

        # For contact force visualization.
        self.contact_results = ContactResults()

    def ExpandQlist(self, q_list):
        q_list_expanded = []
        for q in q_list:
            if isinstance(q, list):
                for qi in q:
                    q_list_expanded.append(qi)
            else:
                q_list_expanded.append(q)

        return q_list_expanded

    def ConcatenatetQlist(self, q_list):
        q_list_concatenated = []
        for q in q_list:
            if isinstance(q, list):
                q_list_concatenated.append(np.concatenate(q))
            else:
                q_list_concatenated.append(q)
        return q_list_concatenated

    def UpdateConfiguration(self, q_list: List):
        """
        :param q_list = [q_u0, q_u1, ..., q_a0, q_a1... q_an].
            A list of np arrays, each array
            is the configuration of a model instance in self.models_list
        :return:
        """
        # Update state in plant_context
        q_list_expanded = self.ExpandQlist(q_list)
        assert len(q_list_expanded) == len(self.models_list_expanded)

        for i in range(len(q_list_expanded)):
            model = self.models_list_expanded[i]
            self.plant.SetPositions(
                self.context_plant, model, q_list_expanded[i])

    def DrawCurrentConfiguration(self):
        # Body poses
        self.viz.DoPublish(self.context_meshcat, [])

        # Contact forces
        self.context_meshcat_contact.FixInputPort(
            self.contact_viz.GetInputPort("contact_results").get_index(),
            AbstractValue.Make(self.contact_results))
        self.contact_viz.DoPublish(self.context_meshcat_contact, [])

    def UpdateNormalAndTangentialJacobianRows(
            self, body, pC_D: np.array, n_W: np.array, d_W: np.array,
            i_c: int, n_di: int, i_f_start: int, position_indices: List[int],
            Jn: np.array, Jf: np.array,
            jacobian_wrt_variable: JacobianWrtVariable):
        """
        Updates corresonding rows of Jn and Jf.
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
        J_WBi = J_WBi[:, position_indices]

        Jn[i_c] = n_W.dot(J_WBi)
        Jf[i_f_start: i_f_start + n_di] = d_W.dot(J_WBi)

    def FindModelInstanceIndexForBody(self, body):
        i_model = None
        for i, body_indices in enumerate(self.body_indices_list):
            if body.index() in body_indices:
                i_model = i
                break
        return i_model

    def CalcContactJacobians(self, contact_detection_tolerance):
        """
        For all contact detected by scene graph, computes Jn and Jf.
        q = [q_u, q_a]
        :param q:
        :return:
        """
        # Evaluate contacts.
        query_object = self.scene_graph.get_query_output_port().Eval(
            self.context_sg)
        signed_distance_pairs = \
            query_object.ComputeSignedDistancePairwiseClosestPoints(
                contact_detection_tolerance)

        n_c = len(signed_distance_pairs)
        n_d = np.full(n_c, self.nd_per_contact)
        n_f = n_d.sum()
        U = np.zeros(n_c)

        phi = np.zeros(n_c)
        Jn_v_list = []
        Jf_v_list = []
        for i, n_v in enumerate(self.n_v_list):
            Jn_v_list.append(np.zeros((n_c, n_v)))
            Jf_v_list.append(np.zeros((n_f, n_v)))

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
            U[i_c] = self.GetFrictionCoefficientFromSignedDistancePair(sdp)
            bodyA = self.GetMbpBodyFromSceneGraphGeometry(sdp.id_A)
            bodyB = self.GetMbpBodyFromSceneGraphGeometry(sdp.id_B)
            X_AFa = self.inspector.GetPoseInFrame(sdp.id_A)
            X_BFb = self.inspector.GetPoseInFrame(sdp.id_B)
            p_AcA_A = X_AFa.multiply(sdp.p_ACa)
            p_BcB_B = X_BFb.multiply(sdp.p_BCb)

            # TODO: it is assumed contact exists only between model
            #  instances, not between bodies within the same model instance.
            i_model_A = self.FindModelInstanceIndexForBody(bodyA)
            i_model_B = self.FindModelInstanceIndexForBody(bodyB)
            is_A_in = i_model_A is not None
            is_B_in = i_model_B is not None

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

                self.UpdateNormalAndTangentialJacobianRows(
                    body=bodyA, pC_D=p_AcA_A, n_W=n_A_W, d_W=d_A_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=self.velocity_indices_list[i_model_A],
                    Jn=Jn_v_list[i_model_A], Jf=Jf_v_list[i_model_A],
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

                self.UpdateNormalAndTangentialJacobianRows(
                    body=bodyB, pC_D=p_BcB_B, n_W=n_B_W, d_W=d_B_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=self.velocity_indices_list[i_model_B],
                    Jn=Jn_v_list[i_model_B], Jf=Jf_v_list[i_model_B],
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

            elif is_B_in:
                n_B_W = -sdp.nhat_BA_W
                d_B_W = CalcTangentVectors(n_B_W, n_d[i_c])

                self.UpdateNormalAndTangentialJacobianRows(
                    body=bodyB, pC_D=p_BcB_B, n_W=n_B_W, d_W=d_B_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=self.velocity_indices_list[i_model_B],
                    Jn=Jn_v_list[i_model_B], Jf=Jf_v_list[i_model_B],
                    jacobian_wrt_variable=JacobianWrtVariable.kV)
            elif is_A_in:
                n_A_W = sdp.nhat_BA_W
                d_A_W = CalcTangentVectors(n_A_W, n_d[i_c])

                self.UpdateNormalAndTangentialJacobianRows(
                    body=bodyA, pC_D=p_AcA_A, n_W=n_A_W, d_W=d_A_W,
                    i_c=i_c, n_di=n_d[i_c], i_f_start=i_f_start,
                    position_indices=self.velocity_indices_list[i_model_A],
                    Jn=Jn_v_list[i_model_A], Jf=Jf_v_list[i_model_A],
                    jacobian_wrt_variable=JacobianWrtVariable.kV)
            else:
                # either A or B is in self.body_indices_list
                raise RuntimeError("At least one body in a contact pair "
                                   "should be in self.body_indices_list.")

            i_f_start += n_d[i_c]

            # TODO: think this through...
            #  PointPairContactInfo stores the contact force for its bodyB.
            #  For convenience, I assign bodyB_index to bodyA_index in the
            #  PointPairContactInfo, which is wrong, but has no bad
            #  consequence because bodyA_index is not used for contact force
            #  visualization anyway.

            # If bodyB is not None, the contact force for bodyB is always shown.
            if is_A_in:
                X_WD = self.plant.EvalBodyPoseInWorld(
                    self.context_plant, bodyA)
                contact_info_list.append(
                    MyContactInfo(bodyA.index(), bodyA.index(),
                                X_WD.multiply(p_AcA_A), n_A_W, d_A_W))
            elif is_B_in:
                X_WD = self.plant.EvalBodyPoseInWorld(
                    self.context_plant, bodyB)
                contact_info_list.append(
                    MyContactInfo(bodyB.index(), bodyB.index(),
                                X_WD.multiply(p_BcB_B), n_B_W, d_B_W))

            else:
                raise RuntimeError("At least one body in a contact pair "
                                   "should be unactuated.")

        return n_c, n_d, n_f, Jn_v_list, Jf_v_list, phi, U, contact_info_list

    def CalcContactResults(self, contact_info_list: List[MyContactInfo],
                           beta: np.array, h: float, n_c: int, n_d: np.array,
                           friction_coefficient: np.array):
        assert len(contact_info_list) == n_c
        contact_results = ContactResults()
        i_f_start = 0
        for i_c, contact_info in enumerate(contact_info_list):
            i_f_end = i_f_start + n_d[i_c]
            beta_i = beta[i_f_start: i_f_end]
            f_normal_W = contact_info.n_W * beta_i.sum() / h
            f_tangential_W = \
                contact_info.dC_W.T.dot(beta_i) * friction_coefficient[i_c] / h
            contact_results.AddContactInfo(
                PointPairContactInfo(
                    contact_info.bodyA_index,
                    contact_info.bodyB_index,
                    f_normal_W + f_tangential_W,
                    contact_info.p_WC_W,
                    0, 0, PenetrationAsPointPair()))

            i_f_start += n_d[i_c]

        return contact_results

    def GetMbpBodyFromSceneGraphGeometry(self, g_id):
        f_id = self.inspector.GetFrameId(g_id)
        return self.plant.GetBodyFromFrameId(f_id)

    def GetPositionIndicesForModel(self, model_instance_index):
        selector = np.arange(self.plant.num_positions())
        return self.plant.GetPositionsFromArray(
            model_instance_index, selector).astype(np.int)

    def GetVelocityIndicesForModel(self, model_instance_index):
        selector = np.arange(self.plant.num_velocities())
        return self.plant.GetVelocitiesFromArray(
            model_instance_index, selector).astype(np.int)

    def GetFrictionCoefficientFromSignedDistancePair(self, sdp):
        props_A = self.inspector.GetProximityProperties(sdp.id_A)
        props_B = self.inspector.GetProximityProperties(sdp.id_B)
        cf_A = props_A.GetProperty("material", "coulomb_friction")
        cf_B = props_B.GetProperty("material", "coulomb_friction")
        cf = CalcContactFrictionFromSurfaceProperties(cf_A, cf_B)
        return cf.static_friction()

    # TODO: tau_u_ext and h should probably come from elsewhere...
    def StepAnitescu(self,
                     q_list: List[np.array],
                     q_a_cmd_list: List[np.array],
                     tau_u_ext_list: List[np.array],
                     h: float, is_planar: bool,
                     contact_detection_tolerance: float):
        """

        :param q_list: [q_u0, q_u1, ..., q_a0, q_a1... q_an]
        :param q_a_cmd_list: same length as q. If a model is not actuated,
            the corresponding entry is set to a zero-length np array.
        :param tau_u_ext_list: same length as q. If a model is actuated,
            the corresponding entry is set to a zero-length np array.
        :param h: simulation time step.
        :param is_planar:
        :param contact_detection_tolerance:
        :return:
        """
        # TODO: remove this ad hoc check to support more general 3D objects.
        if not is_planar:
            for unactuated_model_idx in self.models_unactuated_indices:
                assert self.n_v_list[unactuated_model_idx] == 6
                assert len(q_list[unactuated_model_idx]) == 7

        self.UpdateConfiguration(q_list)
        q_list = self.ConcatenatetQlist(q_list)
        n_c, n_d, n_f, Jn_v_list, Jf_v_list, phi_l, U, contact_info_list = \
            self.CalcContactJacobians(contact_detection_tolerance)
        nv = self.n_v_list.sum()
        n_models = self.n_v_list.size

        prog = mp.MathematicalProgram()

        v_h_list = []
        for i_model, nv_i in enumerate(self.n_v_list):
            # generalized velocity times time step.
            v_h_i = prog.NewContinuousVariables(nv_i, "v_h%d" % i_model)
            v_h_list.append(v_h_i)
        vh = np.concatenate(v_h_list)

        for i_model in self.models_unactuated_indices:
            P_ext_i = tau_u_ext_list[i_model] * h
            prog.AddLinearCost(-P_ext_i, 0, v_h_list[i_model])

        for i_model in self.models_actuated_indices:
            dq_a_cmd = q_a_cmd_list[i_model] - q_list[i_model]
            prog.AddQuadraticCost(
                self.Kq_a * h, -self.Kq_a.dot(dq_a_cmd) * h, v_h_list[i_model])

        Jn = np.zeros((n_c, nv))
        Jf = np.zeros((n_f, nv))

        j_start = 0
        for i_model in range(n_models):
            j_end = j_start + self.n_v_list[i_model]
            Jn[:, j_start: j_end] = Jn_v_list[i_model]
            Jf[:, j_start: j_end] = Jf_v_list[i_model]

            j_start = j_end

        phi_constraints = np.zeros(n_f)
        J = np.zeros_like(Jf)

        j_start = 0
        for i_c in range(n_c):
            for j in range(n_d[i_c]):
                idx = j_start + j
                J[idx] = Jn[i_c] + U[i_c] * Jf[idx]
                phi_constraints[idx] = phi_l[i_c]
            j_start += n_d[i_c]

        constraints = prog.AddLinearConstraint(
            J, -phi_constraints, np.full_like(phi_constraints, np.inf), vh)

        result = self.solver.Solve(prog, None, None)
        assert result.get_solution_result() == mp.SolutionResult.kSolutionFound
        beta = result.GetDualSolution(constraints)
        beta = np.array(beta).squeeze()

        v_h_value_list = []
        for v_h in v_h_list:
            v_h_value_list.append(result.GetSolution(v_h))

        n_u_models = len(self.models_unactuated_indices)
        dq_a_list = v_h_value_list[n_u_models:]

        dq_u_list = []
        for i_model in self.models_unactuated_indices:
            v_u_h_value = v_h_value_list[i_model]
            if is_planar:
                dq_u_list.append(v_u_h_value)
            else:
                q_u = q_list[i_model]
                Q = q_u[:4]  # Quaternion Q_WB
                E = np.array([[-Q[1], Q[0], -Q[3], Q[2]],
                              [-Q[2], Q[3], Q[0], -Q[1]],
                              [-Q[3], -Q[2], Q[1], Q[0]]])

                dq_u = np.zeros(7)
                dq_u[:4] = 0.5 * E.T.dot(v_u_h_value[:3])
                dq_u[4:] = v_u_h_value[3:]
                dq_u_list.append(dq_u)

        # constraint_values = phi_constraints + result.EvalBinding(constraints)
        contact_results = self.CalcContactResults(
            contact_info_list, beta, h, n_c, n_d, U)
        self.contact_results = contact_results
        return dq_u_list, dq_a_list

    def StepConfiguration(self,
                          q_list: List[np.array],
                          dq_u_list: List[np.array],
                          dq_a_list: List[np.array],
                          is_planar: bool):
        """
        Adds the delta of each model state, e.g. dq_u_list[i], to the
            corresponding model configuration in q_list. If q_list[i]
            includes a quaternion, the quaternion (usually the first four
            numbers of a seven-number array) is normalized.
        :param q_list:
            [q_unactuated0, q_unactuated1, ... q_actuated0, q_actuated1, ...]
        :param dq_u_list: [dq_unactuated0, dq_unactuated1, ...]
        :param dq_a_list: [dq_actuated0, dq_actuated1, ...]
        :param is_planar: whether unactuated models has quaternions in their
            configuration.
        :return: None.
        """
        dq_list = dq_u_list + dq_a_list
        for i_model in self.models_unactuated_indices:
            if is_planar:
                q_list[i_model] += dq_list[i_model]
            else:
                q_u = q_list[i_model]
                q_u += dq_list[i_model]
                if q_u.size == 7:
                    q_u[:4] / np.linalg.norm(q_u[:4])  # normalize quaternion

        # TODO: this is hard-coded for IIWA with end effectors
        for i_model in self.models_actuated_indices:
            if isinstance(q_list[i_model], list):
                q_list[i_model][0] += dq_list[i_model][:7]
                q_list[i_model][1] += dq_list[i_model][7:]
            else:
                q_list[i_model] += dq_list[i_model]

