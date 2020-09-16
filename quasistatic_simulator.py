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
class ContactInfo(object):
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
        plant, scene_graph, robot_model, object_model = setup_environment(
            builder, object_sdf_path)
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

        # Get actuated and un-actuated model instances in respective lists?
        self.models_actuated = [robot_model]
        self.models_unactuated = [object_model]
        self.body_indices_actuated = []
        self.body_indices_unactuated = []
        self.position_indices_actuated = []
        self.position_indices_unactuated = []
        self.velocity_indices_unactuated = []

        for model_a in self.models_actuated:
            self.body_indices_actuated.append(plant.GetBodyIndices(model_a))
            self.position_indices_actuated.append(
                self.GetPositionIndicesForModel(model_a))

        for model_u in self.models_unactuated:
            self.body_indices_unactuated.append(plant.GetBodyIndices(model_u))
            self.position_indices_unactuated.append(
                self.GetPositionIndicesForModel(model_u))
            self.velocity_indices_unactuated.append(
                self.GetVelocityIndicesForModel(model_u))

        # compute n_u and n_a
        self.n_a_list = np.array(
            [plant.num_positions(model) for model in self.models_actuated],
            dtype=np.int)
        self.n_a = self.n_a_list.sum()

        self.n_u_q_list = np.array(
            [plant.num_positions(model) for model in self.models_unactuated],
            dtype=np.int)
        self.n_u_q = self.n_u_q_list.sum()

        self.n_u_v_list = np.array(
            [plant.num_velocities(model) for model in self.models_unactuated],
            dtype=np.int)
        self.n_u_v = self.n_u_v_list.sum()

        self.nd_per_contact = nd_per_contact
        assert self.n_a == self.Kq_a.diagonal().size

        # solver
        self.solver = GurobiSolver()
        assert self.solver.available()

        self.contact_results = ContactResults()

    def UpdateConfiguration(self, q):
        """
        :param q = [q_u, q_a]
        :return:
        """
        # Update state in plant_context
        q_u = q[:self.n_u_q]
        q_a = q[self.n_u_q:]
        assert len(self.models_actuated) <= 1
        assert len(self.models_unactuated) <= 1
        for model_a in self.models_actuated:
            self.plant.SetPositions(
                self.context_plant, model_a, q_a)
        for model_u in self.models_unactuated:
            self.plant.SetPositions(
                self.context_plant, model_u, q_u)

    def DrawCurrentConfiguration(self):
        # Body poses
        self.viz.DoPublish(self.context_meshcat, [])

        # Contact forces
        self.context_meshcat_contact.FixInputPort(
            self.contact_viz.GetInputPort("contact_results").get_index(),
            AbstractValue.Make(self.contact_results))
        self.contact_viz.DoPublish(self.context_meshcat_contact, [])

    def UpdateNormalAndTangentialJacobianRows(
            self, body, pC_D, n_W, dC_W, i_c: int, n_di: int,
            i_f_start: int, position_indices, Jn, Jf, jacobian_wrt_variable):
        """
        Updates corresonding rows of Jn and Jf.
        :param body: a RigidBody object that belongs to either
            self.body_indices_actuated or self.body_indices_unactuated.
            D is the body frame of body.
        :param pC_D: contact point in frame D.
        :param n_W: contact normal pointing into body, expressed in W.
        :param dC_W: tangent vectors spanning the tangent plane.
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
        Jf[i_f_start: i_f_start + n_di] = dC_W.dot(J_WBi)

    def FindContactFromSignedDistancePair(
            self, bodyA, bodyB, p_ACa_A, p_BCb_B, nhat_BA_W, body_indices):
        """
        Determine if either of the two bodies (bodyA and bodyB) are in
        body_indices_list. If true, return
            - the body in body_indices_list.
            - the contact point in the body's frame.
            - the contact normal pointing into the body.
        An exception is thrown if both bodyA and bodyB are in body_indices.

        :param bodyA: A RigidBody object containing geometry id_A in sdp.
        :param bodyB: A RigidBody object containing geometry id_B in sdp.
        :param body_indices: A list/set of body indices.
        :return:
        """
        # D: frame of body
        # pC_D: "contact" point for the body expressed in frame D.
        # n_W: contact normal pointing away from the body expressed in world
        #   frame.

        body_D, pC_D, n_W = None, None, None
        is_A_in = bodyA.index() in body_indices
        is_B_in = bodyB.index() in body_indices

        if is_A_in and is_B_in:
            raise RuntimeError("Self collision cannot be handled yet.")

        if is_A_in:
            body_D = bodyA
            # X_DF: Transformation between frame B, the body frame of body,
            #   and F, the frame of the contact geometry where contact is
            #   detected.
            pC_D = p_ACa_A
            n_W = nhat_BA_W
        elif is_B_in:
            body_D = bodyB
            pC_D = p_BCb_B
            n_W = -nhat_BA_W

        return body_D, pC_D, n_W

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
        Jn_u_q = np.zeros((n_c, self.n_u_q))
        Jf_u_q = np.zeros((n_f, self.n_u_q))

        Jn_u_v = np.zeros((n_c, self.n_u_v))
        Jf_u_v = np.zeros((n_f, self.n_u_v))

        # It is assumed here that q_a_dot = v_a.
        Jn_a = np.zeros((n_c, self.n_a))
        Jf_a = np.zeros((n_f, self.n_a))

        contact_info_list = []

        i_f_start = 0
        for i_c, sdp in enumerate(signed_distance_pairs):
            print("contact %i"%i_c)
            print(self.inspector.GetNameByGeometryId(sdp.id_A))
            print(self.inspector.GetNameByGeometryId(sdp.id_B))
            print(self.inspector.GetFrameId(sdp.id_A))
            print(self.inspector.GetFrameId(sdp.id_B))
            print("distance: ", sdp.distance)
            print("")

            phi[i_c] = sdp.distance
            U[i_c] = self.GetFrictionCoefficientFromSignedDistancePair(sdp)
            body1 = self.GetMbpBodyFromSceneGraphGeometry(sdp.id_A)
            body2 = self.GetMbpBodyFromSceneGraphGeometry(sdp.id_B)
            # 1 and 2 denote the body frames of body1 and body2.
            # F is the contact geometry frame relative to the body to which
            # the contact geometry belongs.
            # p_1C1_1 is the coordinates of the "contact" point C1 relative
            # to the body frame 1 expressed in frame 1.
            X_1F = self.inspector.GetPoseInFrame(sdp.id_A)
            X_2F = self.inspector.GetPoseInFrame(sdp.id_B)
            p_1C1_1 = X_1F.multiply(sdp.p_ACa)
            p_2C2_2 = X_2F.multiply(sdp.p_BCb)

            # A: frame of actuated body.
            # U: frame of unactuated body.
            # body_a: actuated body
            # pCa_A: "contact" point for the actuated body expressed in frame A.
            # n_a_W: contact normal pointing away from the actuated body
            #   expressed in world frame.

            # TODO: when a contact pair exists between an unactuated body and
            #  an actuated body, we need dC_a_W = -dC_u_W. In contrast,
            #  if a contact pair contains a body that is neither actuated nor
            #  unactuated, e.g. the ground, we do not need dC_a_W = -dC_u_W.
            #  As CalcTangentVectors(n, nd) != - CalcTangentVectors(-n, nd),
            #  I have to write awkwardly many if statements to make sure that
            #  dC_a_W = -dC_u_W is True. Make this tidier when we move on to
            #  support multiple unactuated bodies.

            body_a, pCa_A, n_a_W = self.FindContactFromSignedDistancePair(
                body1, body2, p_1C1_1, p_2C2_2, sdp.nhat_BA_W,
                self.body_indices_actuated[0])
            if body_a is not None:
                dC_a_W = CalcTangentVectors(n_a_W, n_d[i_c])

            body_u, pCu_U, n_u_W = self.FindContactFromSignedDistancePair(
                body1, body2, p_1C1_1, p_2C2_2, sdp.nhat_BA_W,
                self.body_indices_unactuated[0])
            if body_u is not None:
                dC_u_W = CalcTangentVectors(n_u_W, n_d[i_c])

            if body_a is not None and body_u is not None:
                dC_a_W = -dC_u_W

            if body_a is not None:
                self.UpdateNormalAndTangentialJacobianRows(
                    body=body_a, pC_D=pCa_A, n_W=n_a_W, i_c=i_c, n_di=n_d[i_c],
                    dC_W=dC_a_W, i_f_start=i_f_start,
                    position_indices=self.position_indices_actuated[0],
                    Jn=Jn_a, Jf=Jf_a,
                    jacobian_wrt_variable=JacobianWrtVariable.kQDot)

            if body_u is not None:
                self.UpdateNormalAndTangentialJacobianRows(
                    body=body_u, pC_D=pCu_U, n_W=n_u_W, i_c=i_c, n_di=n_d[i_c],
                    dC_W=dC_u_W, i_f_start=i_f_start,
                    position_indices=self.position_indices_unactuated[0],
                    Jn=Jn_u_q, Jf=Jf_u_q,
                    jacobian_wrt_variable=JacobianWrtVariable.kQDot)

                self.UpdateNormalAndTangentialJacobianRows(
                    body=body_u, pC_D=pCu_U, n_W=n_u_W, i_c=i_c, n_di=n_d[i_c],
                    dC_W=dC_u_W, i_f_start=i_f_start,
                    position_indices=self.velocity_indices_unactuated[0],
                    Jn=Jn_u_v, Jf=Jf_u_v,
                    jacobian_wrt_variable=JacobianWrtVariable.kV)

            i_f_start += n_d[i_c]

            # TODO: think this through...
            #  PointPairContactInfo stores the contact force of its bodyB.
            #  For convenience, I assign bodyB_index to bodyA_index in the
            #  PointPairContactInfo, which is wrong, but has no bad
            #  consequence because bodyA_index is not used for contact force
            #  visualization anyway.
            if body_u is not None:
                X_WD = self.plant.EvalBodyPoseInWorld(
                    self.context_plant, body_u)
                contact_info_list.append(
                    ContactInfo(body_u.index(), body_u.index(),
                                X_WD.multiply(pCu_U), n_u_W, dC_u_W))
            elif body_a is not None:
                X_WD = self.plant.EvalBodyPoseInWorld(
                    self.context_plant, body_a)
                contact_info_list.append(
                    ContactInfo(body_a.index(), body_a.index(),
                                X_WD.multiply(pCa_A), n_a_W, dC_a_W))
            else:
                raise RuntimeError("At least one body in a contact pair "
                                   "should be unactuated.")

        return (n_c, n_d, n_f, Jn_u_q, Jn_u_v, Jn_a, Jf_u_q, Jf_u_v, Jf_a, phi,
                U, contact_info_list)

    def CalcContactResults(self, contact_info_list: List[ContactInfo],
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
    def StepAnitescu(self, q, q_a_cmd, tau_u_ext, h, is_planar,
                     contact_detection_tolerance):
        #TODO: remove this ad hoc check to support more general 3D systems.
        if not is_planar:
            assert self.n_u_q == 7 and self.n_u_v == 6

        self.UpdateConfiguration(q)
        (n_c, n_d, n_f, Jn_u_q, Jn_u_v, Jn_a, Jf_u_q, Jf_u_v, Jf_a, phi_l,
            U, contact_info_list) = self.CalcContactJacobians(
                contact_detection_tolerance)
        dq_a_cmd = q_a_cmd - q[self.n_u_q:]
        q_u = q[:self.n_u_q]

        prog = mp.MathematicalProgram()

        # generalized velocity times time step.
        v_u_h = prog.NewContinuousVariables(self.n_u_v, "v_u_h")
        v_a_h = prog.NewContinuousVariables(self.n_a, "v_a_h")

        P_ext = tau_u_ext * h

        prog.AddQuadraticCost(
            self.Kq_a * h, -self.Kq_a.dot(dq_a_cmd) * h, v_a_h)
        prog.AddLinearCost(-P_ext, 0, v_u_h)

        Jn = np.hstack([Jn_u_v, Jn_a])
        Jf = np.hstack([Jf_u_v, Jf_a])
        J = np.zeros_like(Jf)
        phi_constraints = np.zeros(n_f)

        j_start = 0
        for i in range(n_c):
            for j in range(n_d[i]):
                idx = j_start + j
                J[idx] = Jn[i] + U[i] * Jf[idx]
                phi_constraints[idx] = phi_l[i]
            j_start += n_d[i]

        dv = np.hstack([v_u_h, v_a_h])
        constraints = prog.AddLinearConstraint(
            J, -phi_constraints, np.full_like(phi_constraints, np.inf), dv)

        result = self.solver.Solve(prog, None, None)
        assert result.get_solution_result() == mp.SolutionResult.kSolutionFound
        beta = result.GetDualSolution(constraints)
        beta = np.array(beta).squeeze()
        dv_a_h_value = result.GetSolution(v_a_h)
        dv_u_h_value = result.GetSolution(v_u_h)

        dq_a = dv_a_h_value
        if is_planar:
            dq_u = dv_u_h_value
        else:
            Q = q_u[:4]  # Quaternion Q_WB
            E = np.array([[-Q[1], Q[0], -Q[3], Q[2]],
                          [-Q[2], Q[3], Q[0], -Q[1]],
                          [-Q[3], -Q[2], Q[1], Q[0]]])

            dq_u = np.zeros(7)
            dq_u[:4] = 0.5 * E.T.dot(dv_u_h_value[:3])
            dq_u[4:] = dv_u_h_value[3:]

        constraint_values = phi_constraints + result.EvalBinding(constraints)
        contact_results = self.CalcContactResults(
            contact_info_list, beta, h, n_c, n_d, U)
        self.contact_results = contact_results
        return dq_a, dq_u, beta, constraint_values, result, contact_results
