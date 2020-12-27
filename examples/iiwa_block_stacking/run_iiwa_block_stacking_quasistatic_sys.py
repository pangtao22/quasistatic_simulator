
from setup_environments import create_iiwa_plant_with_schunk
from examples.iiwa_block_stacking.trajectory_generation import *
from examples.iiwa_block_stacking.iiwa_block_stacking_quasistatic_sys import *


#%%
setup_environment = create_iiwa_plant_with_schunk
time_step = 0.3

builder = DiagramBuilder()
q_sys = QuasistaticSystem(
    setup_environment=setup_environment,
    nd_per_contact=4,
    object_sdf_paths=object_sdf_paths,
    joint_stiffness=[Kp_iiwa, Kp_schunk],
    time_step_seconds=time_step)
builder.AddSystem(q_sys)

# visualization
viz = ConnectMeshcatVisualizer(
    builder=builder,
    scene_graph=q_sys.q_sim.scene_graph,
    output_port=q_sys.query_object_output_port,
    draw_period=max(time_step, 1 / 30.))

# trajectory sources.
q_a_traj_list = [q_iiwa_traj, q_schunk_traj]
for model, q_traj in zip(q_sys.q_sim.models_actuated, q_a_traj_list):
    traj_source = TrajectorySource(q_traj)
    builder.AddSystem(traj_source)
    builder.Connect(
        traj_source.get_output_port(0),
        q_sys.get_commanded_positions_input_port(model))

diagram = builder.Build()
RenderSystemWithGraphviz(diagram)

#%% initial conditions.
(model_instance_indices_u,
 model_instance_indices_a) = q_sys.q_sim.get_model_instance_indices()

q0_dict = create_initial_state_dictionary(
    q0_iiwa=q_iiwa_traj.value(0).squeeze(),
    q0_schunk=q_schunk_traj.value(0).squeeze(),
    q_u0_list=q_u0_list,
    model_instance_indices_u=model_instance_indices_u,
    model_instance_indices_a=model_instance_indices_a)


#%% simulation.
sim = Simulator(diagram)
q_sys.set_initial_state(q0_dict)
sim.Initialize()

sim.set_target_realtime_rate(0)
sim.AdvanceTo(q_iiwa_traj.end_time())




