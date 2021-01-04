from pydrake.all import (PiecewisePolynomial, TrajectorySource, Simulator,
                         LogOutput)

from quasistatic_simulation.quasistatic_system import *
from examples.setup_environments import CreateControllerPlantFunction
from iiwa_controller.iiwa_controller.robot_internal_controller import (
    RobotInternalController)


def shift_q_traj_to_start_at_minus_h(q_traj: PiecewisePolynomial, h: float):
    if q_traj.start_time() != 0.:
        q_traj.shiftRight(-q_traj.start_time())
    q_traj.shiftRight(-h)


def create_q0_dict_keyed_by_model_instance_index(
        plant: MultibodyPlant,
        q_dict_str: Dict[str, Union[np.array, PiecewisePolynomial]]
) -> Dict[ModelInstanceIndex, Union[np.array, PiecewisePolynomial]]:
    q_dict = dict()
    for model_name, value in q_dict_str.items():
        model = plant.GetModelInstanceByName(model_name)
        q_dict[model] = value
    return q_dict


def create_q0_dict_keyed_by_string(
        plant: MultibodyPlant,
        q_dict: Dict[ModelInstanceIndex, Union[np.array, PiecewisePolynomial]]
) -> Dict[str, Union[np.array, PiecewisePolynomial]]:
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


def run_quasistatic_sim(
        q_a_traj_dict_str: Dict[str, PiecewisePolynomial],
        q0_dict_str: Dict[str, PiecewisePolynomial],
        Kp_list: List[np.array],
        object_sdf_paths: List[str],
        setup_environment: SetupEnvironmentFunction,
        h: float,
        gravity: np.array,
        is_visualizing: bool,
        real_time_rate: float):

    builder = DiagramBuilder()
    q_sys = QuasistaticSystem(
        setup_environment=setup_environment,
        gravity=gravity,
        nd_per_contact=4,
        object_sdf_paths=object_sdf_paths,
        joint_stiffness=Kp_list,
        time_step_seconds=h)
    builder.AddSystem(q_sys)

    # update dictionaries with ModelInstanceIndex keys.
    q_a_traj_dict = create_q0_dict_keyed_by_model_instance_index(
        q_sys.plant, q_dict_str=q_a_traj_dict_str)
    q0_dict = create_q0_dict_keyed_by_model_instance_index(
        q_sys.plant, q_dict_str=q0_dict_str)

    # trajectory sources.
    assert len(q_sys.q_sim.models_actuated) == len(q_a_traj_dict)
    for model in q_sys.q_sim.models_actuated:
        # Make sure that q_traj start at 0.
        q_traj = q_a_traj_dict[model]
        shift_q_traj_to_start_at_minus_h(q_traj, h)
        traj_source = TrajectorySource(q_traj)
        builder.AddSystem(traj_source)
        builder.Connect(
            traj_source.get_output_port(0),
            q_sys.get_commanded_positions_input_port(model))

    # log states.
    loggers_dict = dict()
    for model in q_sys.q_sim.models_all:
        loggers_dict[model] = LogOutput(
            q_sys.get_state_output_port(model), builder)

    # visualization
    if is_visualizing:
        ConnectMeshcatVisualizer(
            builder=builder,
            scene_graph=q_sys.q_sim.scene_graph,
            output_port=q_sys.query_object_output_port,
            draw_period=max(h, 1 / 30.))

    diagram = builder.Build()

    # Construct simulator and run simulation.
    t_final = find_t_final_from_commanded_trajectories(q_a_traj_dict)
    sim_quasistatic = Simulator(diagram)
    q_sys.set_initial_state(q0_dict)
    sim_quasistatic.Initialize()
    sim_quasistatic.set_target_realtime_rate(real_time_rate)
    sim_quasistatic.AdvanceTo(t_final)

    return create_q0_dict_keyed_by_string(q_sys.plant, loggers_dict), q_sys


def run_mbp_sim(
        q_a_traj: PiecewisePolynomial,
        Kp_a: np.array,
        q0_dict_str: Dict[str, PiecewisePolynomial],
        object_sdf_paths: List[str],
        setup_environment: SetupEnvironmentFunction,
        create_controller_plant: CreateControllerPlantFunction,
        h: float,
        gravity: np.array,
        is_visualizing: bool,
        real_time_rate: float):
    """
    Only supports one actuated model instance, which must have an accompanying
        CreateControllerPlantFunction function.
    """

    builder = DiagramBuilder()
    plant, scene_graph, robot_models, object_models = \
        setup_environment(builder, object_sdf_paths, h, gravity)
    assert len(robot_models) == 1
    robot_model = robot_models[0]

    # controller plant.
    plant_robot, _ = create_controller_plant(gravity)
    controller_robot = RobotInternalController(
        plant_robot=plant_robot, joint_stiffness=Kp_a,
        controller_mode="impedance")
    builder.AddSystem(controller_robot)
    builder.Connect(controller_robot.GetOutputPort("joint_torques"),
                    plant.get_actuation_input_port(robot_model))
    builder.Connect(plant.get_state_output_port(robot_model),
                    controller_robot.robot_state_input_port)

    # robot trajectory source
    shift_q_traj_to_start_at_minus_h(q_a_traj, h)
    traj_source = TrajectorySource(q_a_traj)
    builder.AddSystem(traj_source)
    builder.Connect(
        traj_source.get_output_port(0),
        controller_robot.joint_angle_commanded_input_port)

    # visualization.
    if is_visualizing:
        ConnectMeshcatVisualizer(builder, scene_graph)

    # logs.
    loggers_dict = dict()
    for model in (robot_models + object_models):
        logger = LogOutput(plant.get_state_output_port(model), builder)
        logger.set_publish_period(0.01)
        loggers_dict[model] = logger

    diagram = builder.Build()

    q0_dict = create_q0_dict_keyed_by_model_instance_index(
        plant, q_dict_str=q0_dict_str)

    # Construct simulator and run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()
    context_controller = diagram.GetSubsystemContext(
        controller_robot, context)
    context_plant = diagram.GetSubsystemContext(plant, context)

    controller_robot.tau_feedforward_input_port.FixValue(
        context_controller,
        np.zeros(controller_robot.tau_feedforward_input_port.size()))

    # robot initial configuration.
    # Makes sure that q0_dict has enough initial conditions for every model
    # instance in plant.
    for model, q0 in q0_dict.items():
        plant.SetPositions(context_plant, model, q0)

    sim.Initialize()
    sim.set_target_realtime_rate(real_time_rate)
    sim.AdvanceTo(q_a_traj.end_time())

    return create_q0_dict_keyed_by_string(plant, loggers_dict)
