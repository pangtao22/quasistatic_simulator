import copy

from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.simulator import GradientMode, InternalVisualizationType
from sim_setup import *

# %% Run simulation.
q_parser = QuasistaticParser(os.path.join(models_dir, q_model_path))

q_sim = q_parser.make_simulator_cpp()
q_sim_py = q_parser.make_simulator_py(
    internal_vis=InternalVisualizationType.Python
)

#%%
plant = q_sim.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)
q0_dict = {idx_a: np.array([0.0]), idx_u: np.array([1.0])}
q0 = q_sim.get_q_vec_from_dict(q0_dict)

q_sim_py.update_mbp_positions(q0_dict)
q_sim_py.draw_current_configuration()

#%%
sim_params = copy.deepcopy(q_sim.get_sim_params())
sim_params.unactuated_mass_scale = 10.0
u = np.array([2.0])

for h in [1e-3, 1e-2, 0.1, 0.5]:
    sim_params.h = h
    print(f"h={h}", q_sim.calc_dynamics(q0, u, sim_params))
