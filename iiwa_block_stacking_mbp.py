
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import (
    MeshcatVisualizer, MeshcatContactVisualizer)
from pydrake.systems.primitives import TrajectorySource, LogOutput
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import PidController
from pydrake.all import PiecewisePolynomial

from contact_aware_control.plan_runner.robot_internal_controller import (
    RobotInternalController)
from contact_aware_control.plan_runner.setup_three_link_arm import (
    Create3LinkArmControllerPlant)

from quasistatic_simulator import *
from meshcat_camera_utils import SetOrthographicCameraYZ

from setup_environments import (
    CreateIiwaPlantWithMultipleObjects, create_iiwa_plant_with_schunk,
    box3d_big_sdf_path, box3d_medium_sdf_path, box3d_small_sdf_path,
    box3d_8cm_sdf_path, box3d_7cm_sdf_path)

from contact_aware_control.plan_runner.setup_iiwa import (
    create_iiwa_controller_plant)


def run_sim(q_traj_iiwa: PiecewisePolynomial,
            q_traj_schunk: PiecewisePolynomial,
            Kp_iiwa: np.array,
            Kp_schunk: np.array,
            q_u0: np.array):

    #%%  Build diagram.
    builder = DiagramBuilder()
    plant, scene_graph, robot_models, object_models = \
        create_iiwa_plant_with_schunk(builder, [box3d_8cm_sdf_path])

    iiwa_model, schunk_model = robot_models
    object_model = object_models[0]

    # IIWA controller
    plant_robot, _ = create_iiwa_controller_plant()
    controller_iiwa = RobotInternalController(
        plant_robot=plant_robot, joint_stiffness=Kp_iiwa,
        controller_mode="impedance")
    builder.AddSystem(controller_iiwa)
    builder.Connect(controller_iiwa.GetOutputPort("joint_torques"),
                    plant.get_actuation_input_port(iiwa_model))
    builder.Connect(plant.get_state_output_port(iiwa_model),
                    controller_iiwa.robot_state_input_port)

    # IIWA Trajectory source
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
    traj_source_schunk = TrajectorySource(q_traj_schunk)
    builder.AddSystem(traj_source_schunk)
    builder.Connect(
        traj_source_schunk.get_output_port(0),
        controller_schunk.get_input_port_desired_state())

    # meshcat visualizer
    viz = MeshcatVisualizer(
        scene_graph, frames_to_draw={"three_link_arm": {"link_ee"}})
    builder.AddSystem(viz)
    builder.Connect(
        scene_graph.get_pose_bundle_output_port(),
        viz.GetInputPort("lcm_visualization"))

    # meshcat contact visualizer
    contact_viz = MeshcatContactVisualizer(meshcat_viz=viz, plant=plant)
    builder.AddSystem(contact_viz)
    builder.Connect(
        scene_graph.get_pose_bundle_output_port(),
        contact_viz.GetInputPort("pose_bundle"))
    builder.Connect(
        plant.GetOutputPort("contact_results"),
        contact_viz.GetInputPort("contact_results"))

    diagram = builder.Build()

    #%% Run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()
    context_controller = diagram.GetSubsystemContext(controller_iiwa, context)
    context_plant = diagram.GetSubsystemContext(plant, context)

    controller_iiwa.tau_feedforward_input_port.FixValue(
        context_controller, np.zeros(3))
    plant.SetPositions(context_plant, object_model, q_u0)

    q_iiwa_0 = q_traj_iiwa.value(0).squeeze()
    q_schunk_0 = q_traj_schunk.value(0).squeeze()
    t_final = q_traj_iiwa.end_time()

    plant.SetPositions(context_plant, iiwa_model, q_iiwa_0)
    plant.SetPositions(context_plant, schunk_model, q_schunk_0)

    #%%
    sim.Initialize()
    sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(t_final)


