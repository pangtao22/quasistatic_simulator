import numpy as np
from pydrake.all import LeafSystem, BasicVector
import tqdm

from sim_setup import *


class SimpleTrajectorySource(LeafSystem):
    def __init__(self, q_traj: PiecewisePolynomial):
        super().__init__()
        self.q_traj = q_traj

        self.x_output_port = self.DeclareVectorOutputPort(
            "x", BasicVector(q_traj.rows() * 2), self.calc_x
        )

        self.t_start = 0.0

    def calc_x(self, context, output):
        t = context.get_time() - self.t_start
        q = self.q_traj.value(t).ravel()
        v = self.q_traj.derivative(1).value(t).ravel()
        output.SetFromVector(np.hstack([q, v]))

    def set_t_start(self, t_start_new: float):
        self.t_start = t_start_new


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
    object_sdf_paths=object_sdf_dict,
    time_step=4e-2,  # Only useful for MBP simulations.
    gravity=quasistatic_sim_params.gravity,
)

# robot trajectory source
robot_model = plant.GetModelInstanceByName(robot_name)
q_a_traj = q_a_traj_dict_str[robot_name]

shift_q_traj_to_start_at_minus_h(q_a_traj, h)
traj_source_qv = SimpleTrajectorySource(q_a_traj)
builder.AddSystem(traj_source_qv)

# controller, critically damped PID for ball with mass = 1kg.
pid = PidController(Kp, np.zeros(1), 2 * np.sqrt(Kp))
builder.AddSystem(pid)
builder.Connect(
    pid.get_output_port_control(), plant.get_actuation_input_port(robot_model)
)

builder.Connect(
    plant.get_state_output_port(robot_model),
    pid.get_input_port_estimated_state(),
)

# PID also needs velocity reference.
builder.Connect(traj_source_qv.get_output_port(0), pid.get_input_port_desired_state())

# visulization
meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph)

# logs.
loggers_dict = dict()
for model in robot_models.union(object_models):
    logger = LogOutput(plant.get_state_output_port(model), builder)
    loggers_dict[model] = logger

diagram = builder.Build()


#%% simulation.
n_targets = 1
q_a_targets = np.linspace(0.7, 0.9, n_targets)
q_finals = np.zeros((n_targets, 2))

q0 = np.zeros(2)  # [qa_0, qu_0]
q0[0] = qa_knots[0]
q0[1] = qu0

for i in tqdm.tqdm(range(n_targets)):
    context = diagram.CreateDefaultContext()
    sim = Simulator(diagram, context)

    # initial condition.
    context_plant = plant.GetMyContextFromRoot(context)
    plant.SetPositions(context_plant, q0)

    # Set new qa_traj.
    qa_knots[1] = q_a_targets[i]
    qa_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)
    traj_source_qv.q_traj = qa_traj

    sim.set_target_realtime_rate(1)
    sim.AdvanceTo(q_a_traj.end_time())

    # extract final state
    q_finals[i] = plant.GetPositions(context_plant)


# %%
object_model = plant.GetModelInstanceByName(object_name)
logger_robot = loggers_dict[robot_model]
logger_object = loggers_dict[object_model]

t = logger_robot.sample_times()
x_robot = logger_robot.data()
x_object = logger_object.data()

plt.figure()
plt.plot(t, x_object[0] - x_robot[0] - 0.2)
# plt.ylim([-0.02, 0.02])
plt.grid()
plt.show()


#%%
plt.figure()
plt.plot(q_finals[:, 0], q_finals[:, 1])
plt.show()
