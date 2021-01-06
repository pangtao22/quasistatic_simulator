from pydrake.all import LeafSystem, BasicVector, PortDataType

from .quasistatic_simulator import *


#TODO: replace Lists in the constructor arguments with dictionaries keyed by
# model instance names.

class QuasistaticSystem(LeafSystem):
    def __init__(self,
                 setup_environment: SetupEnvironmentFunction,
                 gravity: np.array,
                 nd_per_contact: int,
                 object_sdf_paths: List[str],
                 joint_stiffness: List[np.array],
                 time_step_seconds: float):
        LeafSystem.__init__(self)
        self.set_name("quasistatic_system")

        # State updates are triggered by publish events.
        self.h = time_step_seconds
        self.DeclarePeriodicDiscreteUpdate(self.h)
        # need at least one state to call self.DoCalcDiscreteVariableUpdates.
        self.DeclareDiscreteState(1)

        # Quasistatic simulator instance.
        self.q_sim = QuasistaticSimulator(
            setup_environment, gravity, nd_per_contact, object_sdf_paths,
            joint_stiffness, internal_vis=False)
        self.plant = self.q_sim.plant

        # output ports for states of unactuated objects and robots (actuated).
        self.state_output_ports = dict()
        for model in self.q_sim.models_all:
            port_name = self.plant.GetModelInstanceName(model) + "_state"
            nq = self.plant.num_positions(model)

            self.state_output_ports[model] = \
                self.DeclareVectorOutputPort(
                    port_name,
                    BasicVector(nq),
                    lambda context, output, model=model:
                        output.SetFromVector(self.copy_model_state_out(model)))

        # query object output port.
        self.query_object_output_port = \
            self.DeclareAbstractOutputPort(
                "query_object",
                lambda: AbstractValue.Make(QueryObject()),
                self.copy_query_object_out)

        # contact resutls oubput port.
        self.contact_results_output_port = \
            self.DeclareAbstractOutputPort(
                "contact_results",
                lambda: AbstractValue.Make(ContactResults()),
                self.copy_contact_results_out)

        # input ports for commanded positions for robots.
        self.commanded_positions_input_ports = dict()
        for model in self.q_sim.models_actuated:
            port_name = self.plant.GetModelInstanceName(model)
            port_name += "_commanded_position"
            nv = self.q_sim.n_v_dict[model]
            self.commanded_positions_input_ports[model] = \
                self.DeclareInputPort(port_name, PortDataType.kVectorValued, nv)

        # input port for externally applied spatial forces.
        self.spatial_force_input_port = self.DeclareAbstractInputPort(
            "applied_spatial_force",
            AbstractValue.Make([ExternallyAppliedSpatialForce()]))

    def get_state_output_port(self, model: ModelInstanceIndex):
        return self.state_output_ports[model]

    def get_commanded_positions_input_port(self, model: ModelInstanceIndex):
        return self.commanded_positions_input_ports[model]

    def copy_model_state_out(self, model: ModelInstanceIndex):
        return self.plant.GetPositions(self.q_sim.context_plant, model)

    def copy_query_object_out(self, context, query_object_abstract_value):
        query_object_abstract_value.set_value(self.q_sim.query_object)

    def copy_contact_results_out(self,  context,
                                 contact_results_abstract_value):
        contact_results_abstract_value.set_value(self.q_sim.contact_results)

    def set_initial_state(self, q0_dict: Dict[ModelInstanceIndex, np.array]):
        self.q_sim.update_configuration(q0_dict)

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        LeafSystem.DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)
        # Commanded positions.
        q_a_cmd_dict = {
            model: self.commanded_positions_input_ports[model].Eval(context)
            for model in self.q_sim.models_actuated}

        # Gravity for unactuated models.
        tau_ext_u_dict = self.q_sim.calc_gravity_for_unactuated_models()

        # Non-contact external spatial forces for actuated models.
        easf_list = []
        if self.spatial_force_input_port.HasValue(context):
            easf_list = self.spatial_force_input_port.Eval(context)

        tau_ext_a_dict = \
            self.q_sim.get_generalized_force_from_external_spatial_force(
                easf_list)
        tau_ext_dict = {**tau_ext_a_dict, **tau_ext_u_dict}

        self.q_sim.step_anitescu(q_a_cmd_dict, tau_ext_dict, self.h,
                                 contact_detection_tolerance=0.005)
