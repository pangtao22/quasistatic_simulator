import os

from qsim.simulator import *
from examples.iiwa_block_stacking.simulation_parameters import *
from examples.setup_simulations import (
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
    """
    "Manual" means calling state update functions in a for loop, without
    relying on drake's system framework to call them.
    :param h:
    :param is_visualizing:
    :return:
    """
    q_sim = QuasistaticSimulator(
        model_directive_path=model_directive_path,
        robot_stiffness_dict=robot_stiffness_dict,
        object_sdf_paths=object_sdf_paths_dict,
        sim_params=quasistatic_sim_params,
        internal_vis=is_visualizing)

    q0_dict = create_dict_keyed_by_model_instance_index(
        q_sim.plant, q0_dict_str)

    #%% show initial configuration before simulation starts.
    q_sim.update_mbp_positions(q0_dict)
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
    phi_threshold = quasistatic_sim_params.contact_detection_tolerance

    for i in range(n_steps):
        t = h * i
        q_a_cmd_dict = {idx_iiwa: q_iiwa_traj.value(t).squeeze(),
                        idx_schunk: q_schunk_traj.value(t).squeeze()}
        tau_ext_dict = q_sim.calc_tau_ext([])
        q_dict = q_sim.step_default(q_a_cmd_dict, tau_ext_dict, h)
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
    # TODO: figure out why.
    q_quasistatic_logs_dict_str1, t_qs = run_quasistatic_sim_manually(
        h=0.2, is_visualizing=False)

    q_quasistatic_logs_dict_str2, _ = run_quasistatic_sim_manually(
        h=0.2, is_visualizing=False)

    for model_name in q0_dict_str.keys():
        q_log1 = q_quasistatic_logs_dict_str1[model_name]
        q_log2 = q_quasistatic_logs_dict_str2[model_name]
        duration = t_qs[-1]
        print(model_name, np.linalg.norm(q_log1 - q_log2) / duration)
