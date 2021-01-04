
from pydrake.all import Simulator

from contact_aware_control.plan_runner.plan_utils import (
    RenderSystemWithGraphviz)

from examples.setup_environments import create_iiwa_plant_with_schunk
from examples.iiwa_block_stacking.trajectory_generation import *
from examples.setup_simulation_diagram import (
    run_quasistatic_sim)

#%%
# Simulation time step.
h = 0.2
gravity = np.array([0, 0, -10.])
diagram, loggers_dict, q_sys = run_quasistatic_sim(
    q_a_traj_dict_str=[q_iiwa_traj, q_schunk_traj],
    Kp_list=[Kp_iiwa, Kp_schunk],
    setup_environment=create_iiwa_plant_with_schunk,
    object_sdf_paths=object_sdf_paths,
    h=h,
    gravity=gravity,
    is_visualizing=True)

#%%
RenderSystemWithGraphviz(diagram)

#%% initial conditions.
(model_instance_indices_u,
 model_instance_indices_a) = q_sys.q_sim.get_model_instance_indices()

t_start = q_iiwa_traj.start_time()
q0_dict = create_initial_state_dictionary(
    q0_iiwa=q_iiwa_traj.value(t_start).squeeze(),
    q0_schunk=q_schunk_traj.value(t_start).squeeze(),
    q_u0_list=q_u0_list,
    model_instance_indices_u=model_instance_indices_u,
    model_instance_indices_a=model_instance_indices_a)


#%% simulation.
sim = Simulator(diagram)
q_sys.set_initial_state(q0_dict)
sim.Initialize()


#%%
sim.set_target_realtime_rate(0)
sim.AdvanceTo(q_iiwa_traj.end_time())
print(sim.get_actual_realtime_rate())

#%%
q_u0_log = loggers_dict[model_instance_indices_u[0]].data().T

