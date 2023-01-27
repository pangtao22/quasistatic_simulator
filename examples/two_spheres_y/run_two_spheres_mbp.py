import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    DiagramBuilder,
    PidController,
    StartMeshcat,
    MeshcatVisualizer,
    LogVectorOutput,
    Simulator_,
    AutoDiffXd,
    InitializeAutoDiff,
    ExtractValue,
    ExtractGradient,
    DiscreteContactSolver
)
import tqdm

from sim_setup import *
from qsim.parser import QuasistaticParser
from qsim.mbp_builder import create_plant_with_robots_and_objects

# %%
q_parser = QuasistaticParser(q_model_path)

builder = DiagramBuilder()
(
    plant,
    scene_graph,
    robot_models,
    object_models,
) = create_plant_with_robots_and_objects(
    builder=builder,
    model_directive_path=q_parser.model_directive_path,
    robot_names=[robot_name],
    object_sdf_paths=q_parser.object_sdf_paths,
    time_step=1e-3,  # Only useful for MBP simulations.
    gravity=q_parser.get_gravity(),
    mbp_solver=DiscreteContactSolver.kSap
)
plant.set_name("MBP")

# robot trajectory source
robot_model = plant.GetModelInstanceByName(robot_name)
object_model = plant.GetModelInstanceByName(object_name)

# controller, critically damped PID for ball with mass = 1kg.
Kp = q_parser.robot_stiffness_dict[robot_name]
pid = PidController(Kp, np.zeros(1), 2 * np.sqrt(Kp))
builder.AddSystem(pid)
builder.Connect(
    pid.get_output_port_control(), plant.get_actuation_input_port(robot_model)
)
pid.set_name("PID")

builder.Connect(
    plant.get_state_output_port(robot_model),
    pid.get_input_port_estimated_state(),
)

# visulization
meshcat = StartMeshcat()
meshcat_vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
meshcat_vis.set_name("meshcat_vis")

# logs.
loggers_dict = dict()
for model in robot_models.union(object_models):
    logger = LogVectorOutput(plant.get_state_output_port(model), builder)
    loggers_dict[model] = logger

diagram = builder.Build()

# %% simulation.
diagram_ad = diagram.ToAutoDiffXd()
plant_ad = diagram_ad.GetSubsystemByName("MBP")
pid_ad = diagram_ad.GetSubsystemByName("PID")
meshcat_vis_ad = diagram_ad.GetSubsystemByName("meshcat_vis")
loggers_ad_dict = {
    model: diagram_ad.GetSubsystemByName(logger.get_name())
    for model, logger in loggers_dict.items()
}

n_targets = 1
q_a_targets = np.linspace(0.7, 0.9, n_targets)
q_finals = np.zeros((n_targets, 2))

q0 = np.zeros(2)  # [qu_0, qa_0]
q0[1] = 0.0
q0[0] = 0.5

for i in tqdm.tqdm(range(n_targets)):
    context = diagram_ad.CreateDefaultContext()
    sim = Simulator_[AutoDiffXd](diagram_ad, context)

    # initial condition.
    context_plant = plant_ad.GetMyContextFromRoot(context)
    plant_ad.SetPositions(context_plant, q0)

    # Set new qa_traj.
    qa_knots[1] = q_a_targets[i]
    qv_desired = np.zeros(2)
    qv_desired[0] = q_a_targets[i]
    context_pid = pid_ad.GetMyContextFromRoot(context)
    pid_ad.GetInputPort(pid.get_input_port_desired_state().get_name()).FixValue(
        context_pid, InitializeAutoDiff(qv_desired)
    )

    sim.set_target_realtime_rate(1)
    meshcat_vis_ad.DeleteRecording()
    meshcat_vis_ad.StartRecording()
    sim.AdvanceTo(4.0)
    meshcat_vis_ad.StopRecording()
    meshcat_vis_ad.PublishRecording()

    # Find logs.
    log_robot = loggers_ad_dict[robot_model].FindLog(context)
    log_object = loggers_ad_dict[object_model].FindLog(context)

    # %%
    t = ExtractValue(log_robot.sample_times())
    x_robot = log_robot.data()
    x_object = log_object.data()

    plt.figure()
    plt.plot(t, ExtractValue(x_object[0]), label="object")
    plt.plot(t, ExtractValue(x_robot[0]), label="robot")
    plt.grid()
    plt.legend()
    plt.show()

    #%%
    plt.figure()
    Dq_object_Du = ExtractGradient(x_object[0])
    plt.plot(t, ExtractValue(x_object[0]), label="q_u")
    plt.plot(t, Dq_object_Du[:, 0], label="Dq_uDu")
    plt.legend()
    plt.grid()
    plt.show()


# %%
