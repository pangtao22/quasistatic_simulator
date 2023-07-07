from typing import Dict

import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
    PortDataType,
    AbstractValue,
    QueryObject,
    ModelInstanceIndex,
    ContactResults,
    ExternallyAppliedSpatialForce,
)

from qsim_cpp import QuasistaticSimulatorCpp

from .simulator import QuasistaticSimulator, QuasistaticSimParameters


class QuasistaticSystem(LeafSystem):
    def __init__(
        self,
        q_sim: QuasistaticSimulatorCpp | QuasistaticSimulator,
        sim_params: QuasistaticSimParameters,
    ):
        """
        A Drake System wrapper for QuasistaticSimulator.
        """
        super().__init__()
        self.set_name("quasistatic_system")

        # Quasistatic simulator instance.
        self.q_sim = q_sim
        self.sim_params = sim_params
        self.plant = self.q_sim.get_plant()
        self.actuated_models = self.q_sim.get_actuated_models()

        # State is defined as q, the configuration of the system.
        self.DeclarePeriodicDiscreteUpdateNoHandler(sim_params.h)
        state_idx = self.DeclareDiscreteState(self.plant.num_positions())
        self.DeclareStateOutputPort("q", state_idx)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=sim_params.h, offset_sec=0.0, update=self.update_q
        )

        self.model_indices_map = self.q_sim.get_position_indices()

        # output ports for states of unactuated objects and robots (actuated).
        self.q_model_output_ports = dict()
        for model in self.q_sim.get_all_models():
            port_name = self.plant.GetModelInstanceName(model) + "_state"
            n_q_model = self.plant.num_positions(model)
            indices_model = self.model_indices_map[model]

            self.q_model_output_ports[model] = self.DeclareVectorOutputPort(
                port_name,
                BasicVector(n_q_model),
                lambda context, output, indices_model=indices_model: (
                    output.SetFromVector(
                        context.get_discrete_state_vector().value()[
                            indices_model
                        ]
                    )
                ),
            )

        # query object output port.
        self.query_object_output_port = self.DeclareAbstractOutputPort(
            "query_object",
            lambda: AbstractValue.Make(QueryObject()),
            self.copy_query_object_out,
        )

        # contact results output port.
        self.contact_results_output_port = self.DeclareAbstractOutputPort(
            "contact_results",
            lambda: AbstractValue.Make(ContactResults()),
            self.copy_contact_results_out,
        )

        # input ports for commanded positions for robots.
        self.commanded_positions_input_ports = dict()
        for model in self.actuated_models:
            port_name = self.plant.GetModelInstanceName(model)
            port_name += "_commanded_position"
            nv = self.plant.num_velocities(model)
            self.commanded_positions_input_ports[
                model
            ] = self.DeclareInputPort(
                port_name, PortDataType.kVectorValued, nv
            )

        # input port for externally applied spatial forces.
        self.spatial_force_input_port = self.DeclareAbstractInputPort(
            "applied_spatial_force",
            AbstractValue.Make([ExternallyAppliedSpatialForce()]),
        )

    def get_q_model_output_port(self, model: ModelInstanceIndex):
        return self.q_model_output_ports[model]

    def get_commanded_positions_input_port(self, model: ModelInstanceIndex):
        return self.commanded_positions_input_ports[model]

    def copy_query_object_out(self, context, query_object_abstract_value):
        query_object_abstract_value.set_value(self.q_sim.get_query_object())

    def copy_contact_results_out(
        self, context, contact_results_abstract_value
    ):
        contact_results_abstract_value.set_value(
            self.q_sim.get_contact_results()
        )

    def set_initial_state(self, q0_dict: Dict[ModelInstanceIndex, np.array]):
        self.q_sim.update_mbp_positions(q0_dict)

    def update_q(self, context, discrete_state):
        # Commanded positions.
        q_a_cmd_dict = {
            model: self.commanded_positions_input_ports[model].Eval(context)
            for model in self.actuated_models
        }

        # Non-contact external spatial forces for actuated models.
        easf_list = []
        if self.spatial_force_input_port.HasValue(context):
            easf_list = self.spatial_force_input_port.Eval(context)

        tau_ext_dict = self.q_sim.calc_tau_ext(easf_list)

        q = context.get_discrete_state_vector()
        self.q_sim.step(q_a_cmd_dict, tau_ext_dict, self.sim_params)
        q_next = self.q_sim.get_mbp_positions_as_vec()
        discrete_state.get_mutable_vector().SetFromVector(q_next)
