from typing import Union
import numpy as np
from pydrake.all import (PiecewisePolynomial, TrajectorySource, Simulator,
                         LogOutput, SpatialForce, BodyIndex, InputPort,
                         Multiplexer, DiagramBuilder, PidController,
                         MultibodyPlant, MeshcatContactVisualizer,
                         ConnectMeshcatVisualizer)

try:
    from ..qsim.system import *
    from ..qsim.utils import create_plant_with_robots_and_objects
except (ImportError, ValueError):
    from qsim.system import *
    from qsim.utils import create_plant_with_robots_and_objects

from robotics_utilities.iiwa_controller.robot_internal_controller import (
    RobotInternalController)


class LoadApplier(LeafSystem):
    def __init__(self, F_WB_traj: PiecewisePolynomial, body_idx: BodyIndex):
        LeafSystem.__init__(self)
        self.set_name("load_applier")

        self.spatial_force_output_port = \
            self.DeclareAbstractOutputPort(
                "external_spatial_force",
                lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
                self.calc_output)

        self.F_WB_traj = F_WB_traj
        self.body_idx = body_idx

    def calc_output(self, context, spatial_forces_vector):
        t = context.get_time()

        easf = ExternallyAppliedSpatialForce()
        F = self.F_WB_traj.value(t).squeeze()
        easf.F_Bq_W = SpatialForce([0, 0, 0], F)
        easf.body_index = self.body_idx

        spatial_forces_vector.set_value([easf])


def shift_q_traj_to_start_at_minus_h(q_traj: PiecewisePolynomial, h: float):
    if q_traj.start_time() != 0.:
        q_traj.shiftRight(-q_traj.start_time())
    q_traj.shiftRight(-h)


def create_dict_keyed_by_model_instance_index(
        plant: MultibodyPlant,
        q_dict_str: Dict[str, Union[np.ndarray, PiecewisePolynomial]]
) -> Dict[ModelInstanceIndex, Union[np.ndarray, PiecewisePolynomial]]:
    q_dict = dict()
    for model_name, value in q_dict_str.items():
        model = plant.GetModelInstanceByName(model_name)
        q_dict[model] = value
    return q_dict


def create_dict_keyed_by_string(
        plant: MultibodyPlant,
        q_dict: Dict[ModelInstanceIndex, Union[np.ndarray, PiecewisePolynomial]]
) -> Dict[str, Union[np.ndarray, PiecewisePolynomial]]:
    q_dict_str = dict()
    for model, value in q_dict.items():
        model_name = plant.GetModelInstanceName(model)
        q_dict_str[model_name] = value
    return q_dict_str


def find_t_final_from_commanded_trajectories(
        q_a_traj_dict: Dict[any, PiecewisePolynomial]):
    t_finals = [q_a_traj.end_time() for q_a_traj in q_a_traj_dict.values()]

    # Make sure that all commanded trajectories have the same length.
    assert all([t_i == t_finals[0] for t_i in t_finals])
    return t_finals[0]


def add_externally_applied_generalized_force(
        builder: DiagramBuilder,
        spatial_force_input_port: InputPort,
        F_WB_traj: PiecewisePolynomial,
        body_idx: BodyIndex):

    load_applier = LoadApplier(F_WB_traj, body_idx)
    builder.AddSystem(load_applier)
    builder.Connect(
        load_applier.spatial_force_output_port, spatial_force_input_port)


def run_quasistatic_sim(
        model_directive_path: str,
        object_sdf_paths: Dict[str, str],
        q_a_traj_dict_str: Dict[str, PiecewisePolynomial],
        q0_dict_str: Dict[str, np.ndarray],
        robot_stiffness_dict: Dict[str, np.ndarray],
        h: float,
        sim_params: QuasistaticSimParameters,
        is_visualizing: bool,
        real_time_rate: float,
        backend: str = "python", **kwargs):

    builder = DiagramBuilder()
    q_sys = QuasistaticSystem(
        time_step=h,
        model_directive_path=model_directive_path,
        robot_stiffness_dict=robot_stiffness_dict,
        object_sdf_paths=object_sdf_paths,
        sim_params=sim_params,
        backend=backend)
    builder.AddSystem(q_sys)

    # update dictionaries with ModelInstanceIndex keys.
    q_a_traj_dict = create_dict_keyed_by_model_instance_index(
        q_sys.plant, q_dict_str=q_a_traj_dict_str)
    q0_dict = create_dict_keyed_by_model_instance_index(
        q_sys.plant, q_dict_str=q0_dict_str)

    # trajectory sources.
    assert len(q_sys.q_sim.get_actuated_models()) == len(q_a_traj_dict)
    for model in q_sys.q_sim.get_actuated_models():
        # Make sure that q_traj start at t=-h.
        q_traj = q_a_traj_dict[model]
        shift_q_traj_to_start_at_minus_h(q_traj, h)
        traj_source = TrajectorySource(q_traj)
        builder.AddSystem(traj_source)
        builder.Connect(
            traj_source.get_output_port(0),
            q_sys.get_commanded_positions_input_port(model))

    # externally applied spatial force.
    # TODO: find a better data structure to pass in external spatial forces.
    if "F_WB_traj" in kwargs.keys():
        input_port = q_sys.spatial_force_input_port
        body_idx = q_sys.plant.GetBodyByName(kwargs["body_name"]).index()
        add_externally_applied_generalized_force(
            builder=builder,
            spatial_force_input_port=input_port,
            F_WB_traj=kwargs["F_WB_traj"],
            body_idx=body_idx)

    # log states.
    loggers_dict = dict()
    for model in q_sys.q_sim.get_all_models():
        loggers_dict[model] = LogOutput(
            q_sys.get_state_output_port(model), builder)

    # visualization
    if is_visualizing:
        meshcat_vis = ConnectMeshcatVisualizer(
            builder=builder,
            scene_graph=q_sys.q_sim.get_scene_graph(),
            output_port=q_sys.query_object_output_port,
            draw_period=max(h, 1 / 30.))

        contact_viz = MeshcatContactVisualizer(meshcat_vis, plant=q_sys.plant)
        builder.AddSystem(contact_viz)
        builder.Connect(q_sys.contact_results_output_port,
                        contact_viz.GetInputPort("contact_results"))

    diagram = builder.Build()
    # RenderSystemWithGraphviz(diagram)

    # Construct simulator and run simulation.
    t_final = find_t_final_from_commanded_trajectories(q_a_traj_dict)
    sim = Simulator(diagram)
    q_sys.set_initial_state(q0_dict)
    sim.Initialize()
    sim.set_target_realtime_rate(real_time_rate)
    if is_visualizing:
        meshcat_vis.reset_recording()
        meshcat_vis.start_recording()

    sim.AdvanceTo(t_final)

    if is_visualizing:
        meshcat_vis.publish_recording()
        res = meshcat_vis.vis.static_html()
        with open("quasistatic_sim.html", "w") as f:
            f.write(res)
    return create_dict_keyed_by_string(q_sys.plant, loggers_dict), q_sys


def run_mbp_sim(
        model_directive_path: str,
        object_sdf_paths: Dict[str, str],
        q_a_traj_dict: Dict[str, PiecewisePolynomial],
        q0_dict_str: Dict[str, np.ndarray],
        robot_stiffness_dict: Dict[str, np.ndarray],
        robot_controller_dict: Dict[str, LeafSystem],
        h: float,
        gravity: np.ndarray,
        is_visualizing: bool,
        real_time_rate: float, **kwargs):
    """
    kwargs is used to handle externally applied spatial forces. Currently
        only supports applying one force (no torque) at the origin of the body
        frame of one body. To apply such forces, kwargs need to have
            - F_WB_traj: trajectory of the force, and
            - body_name: the body to which the force is applied.

    """

    builder = DiagramBuilder()
    plant, scene_graph, robot_models, object_models = \
        create_plant_with_robots_and_objects(
            builder=builder,
            model_directive_path=model_directive_path,
            robot_names=[name for name in robot_stiffness_dict.keys()],
            object_sdf_paths=object_sdf_paths,
            time_step=h,  # Only useful for MBP simulations.
            gravity=gravity)

    # controller.
    for robot_name, joint_stiffness in robot_stiffness_dict.items():
        robot_model = plant.GetModelInstanceByName(robot_name)
        q_a_traj = q_a_traj_dict[robot_name]

        # robot trajectory source
        shift_q_traj_to_start_at_minus_h(q_a_traj, h)
        traj_source_q = TrajectorySource(q_a_traj)
        builder.AddSystem(traj_source_q)

        controller = robot_controller_dict[robot_name]
        builder.AddSystem(controller)

        if isinstance(controller, RobotInternalController):
            builder.Connect(controller.GetOutputPort("joint_torques"),
                            plant.get_actuation_input_port(robot_model))
            builder.Connect(plant.get_state_output_port(robot_model),
                            controller.robot_state_input_port)
            builder.Connect(
                traj_source_q.get_output_port(),
                controller.joint_angle_commanded_input_port)
        elif isinstance(controller, PidController):
            builder.Connect(
                controller.get_output_port_control(),
                plant.get_actuation_input_port(robot_model))
            builder.Connect(
                plant.get_state_output_port(robot_model),
                controller.get_input_port_estimated_state())

            # PID also needs velocity reference.
            v_a_traj = q_a_traj.derivative(1)
            n_q = plant.num_positions(robot_model)
            n_v = plant.num_velocities(robot_model)
            mux = builder.AddSystem(Multiplexer([n_q, n_v]))
            traj_source_v = builder.AddSystem(TrajectorySource(v_a_traj))
            builder.Connect(traj_source_q.get_output_port(),
                            mux.get_input_port(0))
            builder.Connect(traj_source_v.get_output_port(),
                            mux.get_input_port(1))
            builder.Connect(mux.get_output_port(),
                            controller.get_input_port_desired_state())

    # externally applied spatial force.
    if "F_WB_traj" in kwargs.keys():
        input_port = plant.get_applied_spatial_force_input_port()
        body_idx = plant.GetBodyByName(kwargs["body_name"]).index()
        add_externally_applied_generalized_force(
            builder=builder,
            spatial_force_input_port=input_port,
            F_WB_traj=kwargs["F_WB_traj"],
            body_idx=body_idx)

    # visualization.
    if is_visualizing:
        meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph)

    # logs.
    loggers_dict = dict()
    for model in robot_models.union(object_models):
        logger = LogOutput(plant.get_state_output_port(model), builder)
        logger.set_publish_period(0.01)
        loggers_dict[model] = logger

    diagram = builder.Build()

    q0_dict = create_dict_keyed_by_model_instance_index(
        plant, q_dict_str=q0_dict_str)

    # Construct simulator and run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()

    for controller in robot_controller_dict.values():
        if isinstance(controller, RobotInternalController):
            context_controller = diagram.GetSubsystemContext(
                controller, context)
            controller.tau_feedforward_input_port.FixValue(
                context_controller,
                np.zeros(controller.tau_feedforward_input_port.size()))

    # robot initial configuration.
    # Makes sure that q0_dict has enough initial conditions for every model
    # instance in plant.
    context_plant = plant.GetMyContextFromRoot(context)
    for model, q0 in q0_dict.items():
        plant.SetPositions(context_plant, model, q0)

    if is_visualizing:
        meshcat_vis.reset_recording()
        meshcat_vis.start_recording()

    sim.Initialize()

    sim.set_target_realtime_rate(real_time_rate)
    sim.AdvanceTo(q_a_traj.end_time())

    if is_visualizing:
        meshcat_vis.publish_recording()
        res = meshcat_vis.vis.static_html()
        with open("mbp_sim.html", "w") as f:
            f.write(res)

    return create_dict_keyed_by_string(plant, loggers_dict)
