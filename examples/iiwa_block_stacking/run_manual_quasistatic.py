from quasistatic_simulation.quasistatic_simulator import *
from examples.setup_environments import create_iiwa_plant_with_schunk
from examples.iiwa_block_stacking.simulation_parameters import *
from examples.setup_simulation_diagram import (
    shift_q_traj_to_start_at_minus_h,
    create_dict_keyed_by_model_instance_index,
    create_dict_keyed_by_string)


def extract_log_for_object(
        q_log: List[Dict[ModelInstanceIndex, np.array]],
        model: ModelInstanceIndex):
    n = len(q_log)
    m = len(q_log[0][model])
    q_i_log = np.zeros((n, m))
    for t, q_t in enumerate(q_log):
        q_i_log[t] = q_t[model]
    return q_i_log


def run_quasistatic_sim_manually(h: float, is_visualizing: bool):
    q_sim = QuasistaticSimulator(
        create_iiwa_plant_with_schunk,
        gravity=np.array([0, 0, -10.]),
        nd_per_contact=4,
        object_sdf_paths=object_sdf_paths,
        joint_stiffness=Kq_a,
        internal_vis=is_visualizing)

    q0_dict = create_dict_keyed_by_model_instance_index(
        q_sim.plant, q0_dict_str)

    #%% show initial configuration before simulation starts.
    q_sim.update_configuration(q0_dict)
    if is_visualizing:
        q_sim.viz.vis["drake"]["contact_forces"].delete()
        q_sim.draw_current_configuration()

    #%%
    shift_q_traj_to_start_at_minus_h(q_iiwa_traj, h)
    shift_q_traj_to_start_at_minus_h(q_schunk_traj, h)
    n_steps = round(q_iiwa_traj.end_time() / h)
    idx_iiwa, idx_schunk = q_sim.models_actuated

    q_log = [q0_dict]
    t_log = [0.]
    q_a_cmd_log = []

    for i in range(n_steps):
        t = h * i
        q_a_cmd_dict = {idx_iiwa: q_iiwa_traj.value(t).squeeze(),
                        idx_schunk: q_schunk_traj.value(t).squeeze()}
        tau_ext_u_dict = q_sim.calc_gravity_for_unactuated_models()
        tau_ext_a_dict = \
            q_sim.get_generalized_force_from_external_spatial_force([])
        tau_ext_dict = {**tau_ext_a_dict, **tau_ext_u_dict}
        q_dict = q_sim.step_anitescu(q_a_cmd_dict, tau_ext_dict, h,
                                     contact_detection_tolerance=0.005)
        if is_visualizing:
            q_sim.draw_current_configuration()

        q_a_cmd_log.append(q_a_cmd_dict)
        q_log.append(q_dict)
        t_log.append((i + 1) * h)

    q_logs_dict = {
        model: extract_log_for_object(q_log, model)
        for model in q_sim.models_all}

    return (create_dict_keyed_by_string(q_sim.plant, q_logs_dict),
            np.array(t_log))


if __name__ == "__main__":
    # Show that two consecutive simulations have non-deterministic differences.
    q_quasistatic_logs_dict_str1, t_qs = run_quasistatic_sim_manually(
        h=0.2, is_visualizing=False)

    q_quasistatic_logs_dict_str2, _ = run_quasistatic_sim_manually(
        h=0.2, is_visualizing=False)

    for model_name in q0_dict_str.keys():
        q_log1 = q_quasistatic_logs_dict_str1[model_name]
        q_log2 = q_quasistatic_logs_dict_str2[model_name]
        print(model_name, np.linalg.norm(q_log1 - q_log2))
