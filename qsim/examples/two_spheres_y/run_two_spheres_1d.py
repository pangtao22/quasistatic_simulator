import matplotlib.pyplot as plt
from qsim.examples.setup_simulations import run_quasistatic_sim
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.simulator import GradientMode
from sim_setup import *
from qsim_cpp import FiniteDiffGradientCalculator


# %% Run simulation.
q_parser = QuasistaticParser(os.path.join(models_dir, q_model_path))
q_parser.set_sim_params(is_quasi_dynamic=True, h=h)

loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_parser=q_parser,
    backend=QuasistaticSystemBackend.CPP,
    q_a_traj_dict_str={robot_name: qa_traj},
    q0_dict_str=q0_dict_str,
    is_visualizing=True,
    real_time_rate=0.0,
)

# %% construct q and v vectors of MBP from log.
name_to_model_dict = q_sys.q_sim.get_model_instance_name_to_index_map()
logger_qa = loggers_dict_quasistatic_str[robot_name]
logger_qu = loggers_dict_quasistatic_str[object_name]
q_log = np.zeros((T, plant.num_positions()))
v_log = np.zeros((T, plant.num_velocities()))
tau_a_log = np.zeros((T - 1, plant.num_actuated_dofs()))

velocity_indices = q_sys.q_sim.get_velocity_indices()

for name, logger in loggers_dict_quasistatic_str.items():
    model = name_to_model_dict[name]
    for i, j in enumerate(velocity_indices[model]):
        q_log[:, j] = logger.data().T[:, i]

v_log[1:, :] = (q_log[1:, :] - q_log[:-1, :]) / h

Kp = q_parser.get_robot_stiffness_by_name(robot_name)
for l in range(T - 1):
    qa_l = logger_qa.data().T[l]
    qa_l1_cmd = qa_traj.value((l + 1) * h).squeeze()
    tau_a_log[l] = Kp * (qa_l1_cmd - qa_l)

# %% plot q(t).
plt.figure()
plt.title("q(t)")
plt.plot(logger.sample_times(), q_log[:, 0])
plt.plot(logger.sample_times(), q_log[:, 1])
plt.grid(True)
plt.ylabel("t [s]")
plt.show()

# %% compare numerical and analytical gradients
q_sim = q_sys.q_sim
idx_u = name_to_model_dict[object_name]
idx_a = name_to_model_dict[robot_name]
q_u_logger = loggers_dict_quasistatic_str[object_name]
q_a_logger = loggers_dict_quasistatic_str[robot_name]

q_dict = {idx_a: np.array([0.8]), idx_u: np.array([1.0])}
q_nominal = q_sim.get_q_vec_from_dict(q_dict)
u_nominal = [0.82]

# numerical gradients.
fd_gradient_calculator = FiniteDiffGradientCalculator(q_sim)


# kkt active gradients.
sim_params = q_sim.get_sim_params_copy()
sim_params.unactuated_mass_scale = 50
sim_params.gradient_mode = GradientMode.kAB
q_sim.calc_dynamics(q=q_nominal, u=u_nominal, sim_params=sim_params)
dfdu_active = q_sim.get_Dq_nextDqa_cmd()
A_numerical = fd_gradient_calculator.calc_A(
    q_nominal, u_nominal, 1e-3, sim_params
)

print("dfdu_active\n", dfdu_active)
print("A_ad\n", q_sim.get_Dq_nextDq())
print("A_numerical\n", A_numerical)
