import numpy as np

from pydrake.all import (PiecewisePolynomial, TrajectorySource, Simulator,
                         LogOutput, ModelInstanceIndex)

from quasistatic_simulation.quasistatic_system import *
from setup_environments import CreateControllerPlantFunction
from iiwa_controller.iiwa_controller.robot_internal_controller import (
    RobotInternalController)


def shift_q_traj_to_start_at_minus_h(q_traj: PiecewisePolynomial, h: float):
    if q_traj.start_time() != 0.:
        q_traj.shiftRight(-q_traj.start_time())
    q_traj.shiftRight(-h)


def setup_quasistatic_sim_diagram(
        q_a_traj_list: List[PiecewisePolynomial],
        Kp_list: List[np.array],
        object_sdf_paths: List[str],
        setup_environment: SetupEnvironmentFunction,
        h: float,
        gravity: np.array,
        is_visualizing: bool):

    builder = DiagramBuilder()
    q_sys = QuasistaticSystem(
        setup_environment=setup_environment,
        gravity=gravity,
        nd_per_contact=4,
        object_sdf_paths=object_sdf_paths,
        joint_stiffness=Kp_list,
        time_step_seconds=h)
    builder.AddSystem(q_sys)


    # trajectory sources.
    assert len(q_sys.q_sim.models_actuated) == len(q_a_traj_list)
    for model, q_traj in zip(q_sys.q_sim.models_actuated, q_a_traj_list):
        # Make sure that q_traj start at 0.
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

    return diagram, loggers_dict, q_sys


def setup_mbp_sim_diagram(
        q_a_traj: PiecewisePolynomial,
        Kp_a: np.array,
        object_sdf_paths: List[str],
        setup_environment: SetupEnvironmentFunction,
        create_controller_plant: CreateControllerPlantFunction,
        h: float,
        gravity: np.array,
        is_visualizing: bool):
    """
    Only supports one actuated model instance, which must have an accompanying
        CreateControllerPlantFunction function.
    :param q_a_traj_list:
    :param Kp_a:
    :param object_sdf_paths:
    :param setup_environment:
    :param h:
    :param is_visualizing:
    :return:
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

    # initialize simulation.
    diagram = builder.Build()

    return (diagram, plant, controller_robot, loggers_dict, robot_model,
            object_models)


def initialize_mbp_diagram(diagram, plant, controller_robot,
                           q0_dict: Dict[ModelInstanceIndex, np.array]):

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
    # assert plant.num_model_instances() - 2 == len(q0_dict)
    for model, q0 in q0_dict.items():
        plant.SetPositions(context_plant, model, q0)

    return sim


