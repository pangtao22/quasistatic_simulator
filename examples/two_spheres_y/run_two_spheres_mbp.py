from sim_setup import *

# %% create plot for y_u vs y_a
builder = DiagramBuilder()
plant, scene_graph, robot_models, object_models = \
    create_plant_with_robots_and_objects(
        builder=builder,
        model_directive_path=model_directive_path,
        robot_names=[name for name in robot_stiffness_dict.keys()],
        object_sdf_paths=object_sdf_dict,
        time_step=1e-3,  # Only useful for MBP simulations.
        gravity=quasistatic_sim_params.gravity)

# robot trajectory source
robot_model = plant.GetModelInstanceByName(robot_name)
q_a_traj = q_a_traj_dict_str[robot_name]

shift_q_traj_to_start_at_minus_h(q_a_traj, h)
traj_source_q = TrajectorySource(q_a_traj)
builder.AddSystem(traj_source_q)

# controller, critically damped PID for ball with mass = 1kg.
pid = PidController(Kp, np.zeros(1), 2 * np.sqrt(Kp))
builder.AddSystem(pid)
builder.Connect(
    pid.get_output_port_control(),
    plant.get_actuation_input_port(robot_model))

builder.Connect(
    plant.get_state_output_port(robot_model),
    pid.get_input_port_estimated_state())

# PID also needs velocity reference.
v_a_traj = q_a_traj.derivative(1)
mux = builder.AddSystem(Multiplexer([1, 1]))
traj_source_v = builder.AddSystem(TrajectorySource(v_a_traj))
builder.Connect(
    traj_source_q.get_output_port(), mux.get_input_port(0))
builder.Connect(
    traj_source_v.get_output_port(), mux.get_input_port(1))
builder.Connect(mux.get_output_port(), pid.get_input_port_desired_state())

# visulization
meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph)

# logs.
loggers_dict = dict()
for model in robot_models.union(object_models):
    logger = LogOutput(plant.get_state_output_port(model), builder)
    loggers_dict[model] = logger

diagram = builder.Build()
context = diagram.CreateDefaultContext()
sim = Simulator(diagram, context)

# initial condition.
context_plant = plant.GetMyContextFromRoot(context)
plant.SetPositions(context_plant, robot_model, q0_dict_str[robot_name])
for object_model in object_models:
    pass
plant.SetPositions(
    context_plant, object_model, q0_dict_str[object_name])

sim.set_publish_every_time_step(True)
sim.set_target_realtime_rate(1.0)

sim.AdvanceTo(q_a_traj.end_time())

# %%
logger_robot = loggers_dict[robot_model]
logger_object = loggers_dict[object_model]

t = logger_robot.sample_times()
x_robot = logger_robot.data()
x_object = logger_object.data()

plt.figure()
plt.plot(t, x_object[0] - x_robot[0] - 0.2)
plt.ylim([-0.02, 0.02])
plt.grid()
plt.show()

