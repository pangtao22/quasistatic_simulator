from pydrake.all import (LeafSystem, BasicVector, PortDataType, PublishEvent,
    TriggerType)

from .quasistatic_simulator import *


class QuasistaticSystem(LeafSystem):
    def __init__(self,
                 setup_environment: SetupEnvironmentFunction,
                 nd_per_contact: int,
                 object_sdf_paths: List[str],
                 joint_stiffness: List[np.array],
                 time_step_seconds: float):
        LeafSystem.__init__(self)
        self.set_name("quasistatic_system")

        # State updates are triggered by publish events.
        self.DeclarePeriodicPublish(time_step_seconds)
        self.h = time_step_seconds

        # Quasistatic simulator instance.
        self.q_sim = QuasistaticSimulator(
            setup_environment, nd_per_contact, object_sdf_paths,
            joint_stiffness, internal_vis=False)
        self.plant = self.q_sim.plant

        # output ports for states of unactuated objects and robots (actuated).
        self.state_output_ports = dict()
        for model in self.q_sim.models_all:
            port_name = self.plant.GetModelInstanceName(model) + "_state"
            nv = self.q_sim.n_v_dict[model]

            def output(context, y_data):
                y = y_data.get_mutable_value()
                y[:] = self.copy_model_state_out(model)

            self.state_output_ports[model] = \
                self.DeclareVectorOutputPort(port_name, BasicVector(nv), output)

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

        # def on_initialize(context, event):
        #     pass
        #
        # self.DeclareInitializationEvent(
        #     event=PublishEvent(
        #         trigger_type=TriggerType.kInitialization,
        #         callback=on_initialize))

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
        self.q_sim.update_query_object()

    def DoPublish(self, context, event):
        LeafSystem.DoPublish(self, context, event)
        q_a_cmd_dict = {
            model: self.commanded_positions_input_ports[model].Eval(context)
            for model in self.q_sim.models_actuated}

        q_dict = {model: self.copy_model_state_out(model)
                  for model in self.q_sim.models_all}

        tau_ext_dict = self.q_sim.calc_gravity_for_unactuated_models()

        dq_dict = self.q_sim.step_anitescu(
            q_dict, q_a_cmd_dict, tau_ext_dict, self.h,
            is_planar=False,
            contact_detection_tolerance=0.005)

        self.q_sim.step_configuration(q_dict, dq_dict, is_planar=False)
        self.q_sim.update_configuration(q_dict)


