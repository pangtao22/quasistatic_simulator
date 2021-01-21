from pydrake.all import (PiecewisePolynomial, Simulator, PidController,
                         TrajectorySource, LogOutput)

from quasistatic_simulation.quasistatic_simulator import *

from examples.setup_environments import create_iiwa_plant_with_schunk
from examples.setup_simulation_diagram import (
    shift_q_traj_to_start_at_minus_h,
    create_dict_keyed_by_model_instance_index,
    create_dict_keyed_by_string)

from iiwa_controller.iiwa_controller.utils import (
    create_iiwa_controller_plant)
from iiwa_controller.iiwa_controller.robot_internal_controller import (
    RobotInternalController)


def run_mbp_sim(q_traj_iiwa: PiecewisePolynomial,
                x_traj_schunk: PiecewisePolynomial,
                Kp_iiwa: np.array,
                Kp_schunk: np.array,
                object_sdf_paths: List[str],
                q0_dict_str: Dict[str, np.array],
                gravity: np.array,
                time_step: float,
                is_visualizing: bool):

    #%%  Build diagram.
    builder = DiagramBuilder()
    plant, scene_graph, robot_models, object_models = \
        create_iiwa_plant_with_schunk(
            builder, object_sdf_paths, time_step, gravity)

    iiwa_model, schunk_model = robot_models

    # IIWA controller
    gravity = [0, 0, -10.]
    plant_iiwa, _ = create_iiwa_controller_plant(
        gravity, add_schunk_inertia=True)
    controller_iiwa = RobotInternalController(
        plant_robot=plant_iiwa, joint_stiffness=Kp_iiwa,
        controller_mode="impedance")
    builder.AddSystem(controller_iiwa)
    builder.Connect(controller_iiwa.GetOutputPort("joint_torques"),
                    plant.get_actuation_input_port(iiwa_model))
    builder.Connect(plant.get_state_output_port(iiwa_model),
                    controller_iiwa.robot_state_input_port)

    # IIWA Trajectory source
    shift_q_traj_to_start_at_minus_h(q_traj_iiwa, time_step)
    traj_source_iiwa = TrajectorySource(q_traj_iiwa)
    builder.AddSystem(traj_source_iiwa)
    builder.Connect(
        traj_source_iiwa.get_output_port(0),
        controller_iiwa.joint_angle_commanded_input_port)

    # Schunk controller
    controller_schunk = PidController(
        Kp_schunk, np.zeros(2), 2 * 0.7 * np.sqrt(Kp_schunk))

    builder.AddSystem(controller_schunk)
    builder.Connect(
        controller_schunk.get_output_port_control(),
        plant.get_actuation_input_port(schunk_model))
    builder.Connect(
        plant.get_state_output_port(schunk_model),
        controller_schunk.get_input_port_estimated_state())

    # Schunk Trajectory source
    shift_q_traj_to_start_at_minus_h(x_traj_schunk, time_step)
    traj_source_schunk = TrajectorySource(x_traj_schunk)
    builder.AddSystem(traj_source_schunk)
    builder.Connect(
        traj_source_schunk.get_output_port(0),
        controller_schunk.get_input_port_desired_state())

    # meshcat visualizer
    if is_visualizing:
        meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph)

    # meshcat contact visualizer
    # contact_viz = MeshcatContactVisualizer(meshcat_viz=viz, plant=plant)
    # builder.AddSystem(contact_viz)
    # builder.Connect(
    #     scene_graph.get_pose_bundle_output_port(),
    #     contact_viz.GetInputPort("pose_bundle"))
    # builder.Connect(
    #     plant.GetOutputPort("contact_results"),
    #     contact_viz.GetInputPort("contact_results"))

    q0_dict = create_dict_keyed_by_model_instance_index(
        plant, q0_dict_str)

    # Logs
    loggers_dict = dict()
    for model in q0_dict.keys():
        logger = LogOutput(
            plant.get_state_output_port(model), builder)
        loggers_dict[model] = logger
        logger.set_publish_period(0.01)

    diagram = builder.Build()

    #%% Run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()
    context_controller = diagram.GetSubsystemContext(controller_iiwa, context)
    context_plant = diagram.GetSubsystemContext(plant, context)

    controller_iiwa.tau_feedforward_input_port.FixValue(
        context_controller, np.zeros(7))

    for model, q0 in q0_dict.items():
        plant.SetPositions(context_plant, model, q0)

    t_final = q_traj_iiwa.end_time()

    #%%
    if is_visualizing:
        meshcat_vis.reset_recording()
        meshcat_vis.start_recording()

    sim.Initialize()
    sim.set_target_realtime_rate(0.0)
    try:
        sim.AdvanceTo(t_final)
        if is_visualizing:
            meshcat_vis.publish_recording()
            res = meshcat_vis.vis.static_html()
            with open("mbp_sim.html", "w") as f:
                f.write(res)
    except RuntimeError as err:
        print(err)
    finally:
        return create_dict_keyed_by_string(plant, loggers_dict)

