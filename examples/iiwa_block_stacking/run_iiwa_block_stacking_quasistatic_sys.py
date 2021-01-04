
from contact_aware_control.plan_runner.plan_utils import (
    RenderSystemWithGraphviz)

from examples.setup_environments import create_iiwa_plant_with_schunk
from examples.iiwa_block_stacking.simulation_parameters import *
from examples.setup_simulation_diagram import (
    run_quasistatic_sim)

#%%
# Simulation time step.
h = 0.2
loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    Kp_list=[Kp_iiwa, Kp_schunk],
    setup_environment=create_iiwa_plant_with_schunk,
    object_sdf_paths=object_sdf_paths,
    h=h,
    gravity=gravity,
    is_visualizing=True,
    real_time_rate=0.0)

