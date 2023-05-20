from typing import Dict
import enum

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


class QuasistaticSystemBackend(enum.Enum):
    PYTHON = enum.auto()
    CPP = enum.auto()


class QuasistaticSystem(LeafSystem):
    def __init__(
        self,
        model_directive_path: str,
        robot_stiffness_dict: Dict[str, np.ndarray],
        object_sdf_paths: Dict[str, str],
        sim_params: QuasistaticSimParameters,
        backend=QuasistaticSystemBackend.PYTHON,
    ):
        """
        Also refer to the docstring of quasistatic simulator.
        :param time_step:  quasistatic simulation time step in seconds.
        :param model_directive_path:
        :param robot_stiffness_dict:
        :param object_sdf_paths:
        :param sim_params:
        """
        LeafSystem.__init__(self)
        self.set_name("quasistatic_system")

        # State updates are triggered by publish events.
        self.DeclarePeriodicDiscreteUpdateNoHandler(sim_params.h)
        # need at least one state to call self.DoCalcDiscreteVariableUpdates.
        self.DeclareDiscreteState(1)

        # Quasistatic simulator instance.
        if backend == QuasistaticSystemBackend.CPP:
            self.q_sim = QuasistaticSimulatorCpp(
                model_directive_path=model_directive_path,
                robot_stiffness_str=robot_stiffness_dict,
                object_sdf_paths=object_sdf_paths,
                sim_params=sim_params,
            )
        elif backend == QuasistaticSystemBackend.PYTHON:
            self.q_sim = QuasistaticSimulator(
                model_directive_path=model_directive_path,
                robot_stiffness_dict=robot_stiffness_dict,
                object_sdf_paths=object_sdf_paths,
                sim_params=sim_params,
                internal_vis=False,
            )
        else:
            raise RuntimeError(
                "QuasistaticSystem backend must be either python or cpp."
            )

        self.plant = self.q_sim.get_plant()
        self.actuated_models = self.q_sim.get_actuated_models()

        # output ports for states of unactuated objects and robots (actuated).
        self.state_output_ports = dict()
        for model in self.q_sim.get_all_models():
            port_name = self.plant.GetModelInstanceName(model) + "_state"
            nq = self.plant.num_positions(model)

            self.state_output_ports[model] = self.DeclareVectorOutputPort(
                port_name,
                BasicVector(nq),
                lambda context, output, model=model: output.SetFromVector(
                    self.q_sim.get_positions(model)
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
            self.commanded_positions_input_ports[model] = self.DeclareInputPort(
                port_name, PortDataType.kVectorValued, nv
            )

        # input port for externally applied spatial forces.
        self.spatial_force_input_port = self.DeclareAbstractInputPort(
            "applied_spatial_force",
            AbstractValue.Make([ExternallyAppliedSpatialForce()]),
        )

    def get_state_output_port(self, model: ModelInstanceIndex):
        return self.state_output_ports[model]

    def get_commanded_positions_input_port(self, model: ModelInstanceIndex):
        return self.commanded_positions_input_ports[model]

    def copy_query_object_out(self, context, query_object_abstract_value):
        query_object_abstract_value.set_value(self.q_sim.get_query_object())

    def copy_contact_results_out(self, context, contact_results_abstract_value):
        contact_results_abstract_value.set_value(self.q_sim.get_contact_results())

    def set_initial_state(self, q0_dict: Dict[ModelInstanceIndex, np.array]):
        self.q_sim.update_mbp_positions(q0_dict)

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        super().DoCalcDiscreteVariableUpdates(context, events, discrete_state)
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

        self.q_sim.step_default(q_a_cmd_dict, tau_ext_dict)
