
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import (
    MeshcatVisualizer, MeshcatContactVisualizer)
from pydrake.systems.primitives import TrajectorySource
from pydrake.systems.analysis import Simulator

from contact_aware_control.plan_runner.robot_internal_controller import (
    RobotInternalController)
from contact_aware_control.plan_runner.setup_three_link_arm import (
    Create3LinkArmControllerPlant)

from setup_environments import *
from sim_params_3link_arm import *

#%%
object_sdf_path = os.path.join("models", "box_yz_rotation_big.sdf")
q_a0 = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2])
q_u0 = np.array([1.7, 0.5, 0])
q0 = np.hstack([q_u0, q_a0])



#%%  Build diagram.
builder = DiagramBuilder()
plant, scene_graph, robot_model, object_model = \
    CreatePlantFor2dArmWithObject(builder, object_sdf_path)

# robot controller
plant_robot = Create3LinkArmControllerPlant()
controller = RobotInternalController(
    plant_robot=plant_robot, joint_stiffness=Kq_a,
    controller_mode="impedance")
builder.AddSystem(controller)
builder.Connect(controller.GetOutputPort("joint_torques"),
                plant.get_actuation_input_port())
builder.Connect(plant.get_state_output_port(robot_model),
                controller.robot_state_input_port)

# Trajectory source
traj_source = TrajectorySource(q_a_traj)
builder.AddSystem(traj_source)
builder.Connect(
    traj_source.get_output_port(0), controller.joint_angle_commanded_input_port)

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
context_controller = diagram.GetSubsystemContext(controller, context)
context_plant = diagram.GetSubsystemContext(plant, context)

controller.tau_feedforward_input_port.FixValue(context_controller, np.zeros(3))
plant.SetPositions(context_plant, object_model, q_u0)
plant.SetPositions(context_plant, robot_model, q_a0)


#%%
sim.Initialize()
sim.set_target_realtime_rate(0.5)
sim.AdvanceTo(t_final)


