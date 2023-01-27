import unittest
from typing import Union

from pydrake.all import (
    PiecewisePolynomial,
    TrajectorySource,
    Simulator,
    VectorLogSink,
    LogVectorOutput,
    SpatialForce,
    BodyIndex,
    InputPort,
    Multiplexer,
    DiagramBuilder,
    PidController,
    MultibodyPlant,
    MeshcatVisualizer,
    ContactVisualizer,
    StartMeshcat,
    Meshcat,
    DiscreteContactSolver,
    InverseDynamics,
)
from robotics_utilities.iiwa_controller.robot_internal_controller import (
    RobotInternalController,
)
from qsim.parser import QuasistaticParser
from qsim.system import *
from qsim.mbp_builder import create_plant_with_robots_and_objects

# This is an experimental feature pending Drake PR #17674.
try:
    from pydrake.all import PdControllerGains
except ImportError:
    PdControllerGains = None


class LoadApplier(LeafSystem):
    def __init__(self, F_WB_traj: PiecewisePolynomial, body_idx: BodyIndex):
        LeafSystem.__init__(self)
        self.set_name("load_applier")

        self.spatial_force_output_port = self.DeclareAbstractOutputPort(
            "external_spatial_force",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self.calc_output,
        )

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
    if q_traj.start_time() != 0.0:
        q_traj.shiftRight(-q_traj.start_time())
    q_traj.shiftRight(-h)


def create_dict_keyed_by_model_instance_index(
    plant: MultibodyPlant,
    q_dict_str: Dict[str, Union[np.ndarray, PiecewisePolynomial]],
) -> Dict[ModelInstanceIndex, Union[np.ndarray, PiecewisePolynomial]]:
    q_dict = dict()
    for model_name, value in q_dict_str.items():
        model = plant.GetModelInstanceByName(model_name)
        q_dict[model] = value
    return q_dict


def create_dict_keyed_by_string(
    plant: MultibodyPlant,
    q_dict: Dict[ModelInstanceIndex, Union[np.ndarray, PiecewisePolynomial]],
) -> Dict[str, Union[np.ndarray, PiecewisePolynomial]]:
    q_dict_str = dict()
    for model, value in q_dict.items():
        model_name = plant.GetModelInstanceName(model)
        q_dict_str[model_name] = value
    return q_dict_str


def find_t_final_from_commanded_trajectories(
    q_a_traj_dict: Dict[any, PiecewisePolynomial]
):
    t_finals = [q_a_traj.end_time() for q_a_traj in q_a_traj_dict.values()]

    # Make sure that all commanded trajectories have the same length.
    assert all([t_i == t_finals[0] for t_i in t_finals])
    return t_finals[0]


def add_externally_applied_generalized_force(
    builder: DiagramBuilder,
    spatial_force_input_port: InputPort,
    F_WB_traj: PiecewisePolynomial,
    body_idx: BodyIndex,
):
    load_applier = LoadApplier(F_WB_traj, body_idx)
    builder.AddSystem(load_applier)
    builder.Connect(
        load_applier.spatial_force_output_port, spatial_force_input_port
    )


def get_logs_from_sim(
    log_sinks_dict: Dict[ModelInstanceIndex, VectorLogSink], sim: Simulator
):
    loggers_dict = dict()

    context = sim.get_context()
    for model, log_sink in log_sinks_dict.items():
        loggers_dict[model] = log_sink.GetLog(
            log_sink.GetMyContextFromRoot(context)
        )
    return loggers_dict


def run_quasistatic_sim(
    q_parser: QuasistaticParser,
    backend: QuasistaticSystemBackend,
    q_a_traj_dict_str: Dict[str, PiecewisePolynomial],
    q0_dict_str: Dict[str, np.ndarray],
    is_visualizing: bool,
    real_time_rate: float,
    meshcat: Meshcat = None,
    **kwargs,
):
    h = q_parser.get_param_attribute("h")
    builder = DiagramBuilder()
    q_sys = q_parser.make_system(backend=backend)
    builder.AddSystem(q_sys)

    # update dictionaries with ModelInstanceIndex keys.
    q_a_traj_dict = create_dict_keyed_by_model_instance_index(
        q_sys.plant, q_dict_str=q_a_traj_dict_str
    )
    q0_dict = create_dict_keyed_by_model_instance_index(
        q_sys.plant, q_dict_str=q0_dict_str
    )

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
            q_sys.get_commanded_positions_input_port(model),
        )

    # externally applied spatial force.
    # TODO: find a better data structure to pass in external spatial forces.
    if "F_WB_traj" in kwargs.keys():
        input_port = q_sys.spatial_force_input_port
        body_idx = q_sys.plant.GetBodyByName(kwargs["body_name"]).index()
        add_externally_applied_generalized_force(
            builder=builder,
            spatial_force_input_port=input_port,
            F_WB_traj=kwargs["F_WB_traj"],
            body_idx=body_idx,
        )

    # log states.
    log_sinks_dict = dict()
    for model in q_sys.q_sim.get_all_models():
        log_sinks_dict[model] = LogVectorOutput(
            q_sys.get_state_output_port(model), builder
        )

    # visualization
    if is_visualizing:
        if meshcat is None:
            meshcat = StartMeshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, q_sys.query_object_output_port, meshcat
        )
        ContactVisualizer.AddToBuilder(
            builder,
            q_sys.contact_results_output_port,
            meshcat,
        )

    diagram = builder.Build()
    # RenderSystemWithGraphviz(diagram)

    # Construct simulator and run simulation.
    t_final = find_t_final_from_commanded_trajectories(q_a_traj_dict)
    sim = Simulator(diagram)
    q_sys.set_initial_state(q0_dict)
    sim.Initialize()
    sim.set_target_realtime_rate(real_time_rate)
    if is_visualizing:
        meshcat_vis.DeleteRecording()
        meshcat_vis.StartRecording()

    sim.AdvanceTo(t_final)

    # get logs from sim context.
    if is_visualizing:
        meshcat_vis.PublishRecording()
        # res = meshcat_vis.vis.static_html()
        # with open("quasistatic_sim.html", "w") as f:
        #     f.write(res)

    loggers_dict = get_logs_from_sim(log_sinks_dict, sim)
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
    real_time_rate: float,
    meshcat: Meshcat = None,
    mbp_solver: DiscreteContactSolver = DiscreteContactSolver.kTamsi,
    use_implicit_pd_controller: bool = False,
    **kwargs,
):
    """
    robot_controller_dict is keyed by the model instance name of the robots.
    There are three kinds of controllers:
    1. RobotInternalController: an impedance controller implemented by me.
    2. PidController: drake's PID controller.
    3. InverseDynamicsController: it is only used for gravity compensation.
        PD control is implemented implicitly as part of MBP. Note that
        implicit PD controller is an experimental feature in drake PR #17674.
    kwargs is used to handle
     (1) externally applied spatial forces. Currently
        only supports applying one force (no torque) at the origin of the body
        frame of one body. To apply such forces, kwargs need to have
            - F_WB_traj: trajectory of the force, and
            - body_name: the body to which the force is applied.
     (2) robot_damping_dict: Dict[str, np.ndarray]
    """
    robot_damping_dict = (
        kwargs["robot_damping_dict"] if "robot_damping_dict" in kwargs else None
    )

    builder = DiagramBuilder()
    (
        plant,
        scene_graph,
        robot_models,
        object_models,
    ) = create_plant_with_robots_and_objects(
        builder=builder,
        model_directive_path=model_directive_path,
        robot_names=[name for name in robot_stiffness_dict.keys()],
        object_sdf_paths=object_sdf_paths,
        time_step=h,  # Only useful for MBP simulations.
        gravity=gravity,
        mbp_solver=mbp_solver,
        add_robot_pd_controller=use_implicit_pd_controller,
        robot_stiffness_dict=robot_stiffness_dict,
        robot_damping_dict=robot_damping_dict,
    )

    # controller.
    for robot_name, joint_stiffness in robot_stiffness_dict.items():
        robot_model = plant.GetModelInstanceByName(robot_name)
        q_a_traj = q_a_traj_dict[robot_name]

        # robot trajectory source
        shift_q_traj_to_start_at_minus_h(q_a_traj, h)
        controller = robot_controller_dict[robot_name]
        if isinstance(controller, LeafSystem):
            builder.AddSystem(controller)

        if isinstance(controller, RobotInternalController):
            traj_source_q = TrajectorySource(q_a_traj)
            builder.AddSystem(traj_source_q)

            builder.Connect(
                controller.GetOutputPort("joint_torques"),
                plant.get_actuation_input_port(robot_model),
            )
            builder.Connect(
                plant.get_state_output_port(robot_model),
                controller.robot_state_input_port,
            )
            builder.Connect(
                traj_source_q.get_output_port(),
                controller.joint_angle_commanded_input_port,
            )
        elif isinstance(controller, PidController):
            traj_source_q_v = TrajectorySource(
                q_a_traj, output_derivative_order=1
            )
            builder.AddSystem(traj_source_q_v)

            builder.Connect(
                controller.get_output_port_control(),
                plant.get_actuation_input_port(robot_model),
            )
            builder.Connect(
                plant.get_state_output_port(robot_model),
                controller.get_input_port_estimated_state(),
            )

            builder.Connect(
                traj_source_q_v.get_output_port(),
                controller.get_input_port_desired_state(),
            )
        elif isinstance(controller, InverseDynamics):
            # When using implicit PD controller with gravity compensation.

            traj_source_q_v = TrajectorySource(
                q_a_traj, output_derivative_order=1
            )
            builder.AddSystem(traj_source_q_v)

            # InverseDynamics should be in gravity compensation mode.
            assert controller.is_pure_gravity_compensation()
            # controller.get_input_port_estimated_state() is not bound.
            builder.Connect(
                plant.get_state_output_port(robot_model),
                controller.GetInputPort("u0"),
            )
            # controller.get_output_port_force() is not bound.
            builder.Connect(
                controller.GetOutputPort("y0"),
                plant.get_actuation_input_port(robot_model),
            )

            builder.Connect(
                traj_source_q_v.get_output_port(),
                plant.get_desired_state_input_port(robot_model),
            )
        elif controller is None:
            # When using implicit PD controller without gravity compensation.
            traj_source_q_v = TrajectorySource(
                q_a_traj, output_derivative_order=1
            )
            builder.AddSystem(traj_source_q_v)
            builder.Connect(
                traj_source_q_v.get_output_port(),
                plant.get_desired_state_input_port(robot_model),
            )
        else:
            raise RuntimeError("Wrong robot controller type.")

    # externally applied spatial force.
    if "F_WB_traj" in kwargs:
        input_port = plant.get_applied_spatial_force_input_port()
        body_idx = plant.GetBodyByName(kwargs["body_name"]).index()
        add_externally_applied_generalized_force(
            builder=builder,
            spatial_force_input_port=input_port,
            F_WB_traj=kwargs["F_WB_traj"],
            body_idx=body_idx,
        )

    # visualization.
    if is_visualizing:
        if meshcat is None:
            meshcat = StartMeshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat
        )
        ContactVisualizer.AddToBuilder(
            builder,
            plant,
            meshcat,
        )

    # logs.
    log_sinks_dict = dict()
    for model in robot_models.union(object_models):
        logger = LogVectorOutput(
            plant.get_state_output_port(model), builder, 0.01
        )
        log_sinks_dict[model] = logger

    diagram = builder.Build()

    q0_dict = create_dict_keyed_by_model_instance_index(
        plant, q_dict_str=q0_dict_str
    )

    # Construct simulator and run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()

    for controller in robot_controller_dict.values():
        if isinstance(controller, RobotInternalController):
            context_controller = diagram.GetSubsystemContext(
                controller, context
            )
            controller.tau_feedforward_input_port.FixValue(
                context_controller,
                np.zeros(controller.tau_feedforward_input_port.size()),
            )

    # robot initial configuration.
    # Makes sure that q0_dict has enough initial conditions for every model
    # instance in plant.
    context_plant = plant.GetMyContextFromRoot(context)
    for model, q0 in q0_dict.items():
        plant.SetPositions(context_plant, model, q0)

    if is_visualizing:
        meshcat_vis.DeleteRecording()
        meshcat_vis.StartRecording()

    sim.Initialize()

    sim.set_target_realtime_rate(real_time_rate)
    sim.AdvanceTo(q_a_traj.end_time())

    if is_visualizing:
        meshcat_vis.PublishRecording()

    loggers_dict = get_logs_from_sim(log_sinks_dict, sim)

    # diagram structure visualization.
    # from graphviz import Source
    #
    # src = Source(diagram.GetGraphvizString())
    # src.render("system_view.gz", view=False)

    return create_dict_keyed_by_string(plant, loggers_dict)


def compare_q_sim_cpp_vs_py(
    test_case: unittest.TestCase,
    q_parser: QuasistaticParser,
    q_a_traj_dict_str: Dict[str, PiecewisePolynomial],
    q0_dict_str: Dict[str, np.ndarray],
    atol: float,
):
    """
    This function calls run_quasistatic_sim using both the CPP and PYTHON
        backends and makes sure that the logs are close.
    """
    loggers_dict_quasistatic_str_cpp, q_sys_cpp = run_quasistatic_sim(
        q_parser=q_parser,
        backend=QuasistaticSystemBackend.CPP,
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        is_visualizing=False,
        real_time_rate=0.0,
    )

    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_parser=q_parser,
        backend=QuasistaticSystemBackend.PYTHON,
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        is_visualizing=False,
        real_time_rate=0.0,
    )

    for name in loggers_dict_quasistatic_str_cpp.keys():
        q_log_cpp = loggers_dict_quasistatic_str_cpp[name].data()
        q_log = loggers_dict_quasistatic_str[name].data()

        test_case.assertEqual(q_log.shape, q_log_cpp.shape)
        np.testing.assert_allclose(q_log, q_log_cpp, atol=atol)
