import unittest
import sys
from typing import Union, Dict

import numpy as np
from examples.planar_hand_ball.run_planar_hand import (model_directive_path,
                                                       robot_stiffness_dict,
                                                       object_sdf_dict,
                                                       q0_dict_str)
from examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)
from pydrake.all import ModelInstanceIndex
from quasistatic_simulator.qsim.simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.qsim.system import cpp_params_from_py_params

from qsim_cpp import (QuasistaticSimParametersCpp, QuasistaticSimulatorCpp)


def simulate(sim: Union[QuasistaticSimulator, QuasistaticSimulatorCpp],
             q0_dict: Dict[ModelInstanceIndex, np.ndarray], h: float,
             T: int):
    q_dict = {model: np.array(q_model)
              for model, q_model in q0_dict.items()}
    q_dict_log = [{model: np.array(q_model)
                   for model, q_model in q_dict.items()}]
    Dq_nextDq_log = []
    Dq_nextDqa_cmd_log = []
    sim.update_mbp_positions(q_dict)

    for _ in range(T):
        tau_ext_dict = sim.calc_tau_ext([])
        sim.step_default(q0_dict, tau_ext_dict, h)
        q_dict_log.append(sim.get_mbp_positions())
        Dq_nextDq_log.append(sim.get_Dq_nextDq())
        Dq_nextDqa_cmd_log.append(sim.get_Dq_nextDqa_cmd())

    return q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log


class TestPlanarHandBall(unittest.TestCase):
    def test_cpp_vs_python(self):
        """
        Requires python bindings of QuiasistaticSimulator be on the PYTHONPATH.
        Test 3 cases for gradient computation:
        1. from all constraints, cpp vs. python.
        2. from active constraints, cpp vs. python.
        3. cpp from active constraints, python from all constraints.
        """
        h = 0.05
        T = int(round(2 / h))  # num of time steps to simulate forward.

        grad_active_py_list = [False, True, False]
        grad_active_cpp_list = [False, True, True]
        atol_list = [1e-9, 1e-9, 1e-4]

        for grad_active_py, grad_active_cpp, atol in zip(grad_active_py_list,
                                                         grad_active_cpp_list,
                                                         atol_list):
            sim_params = QuasistaticSimParameters(
                gravity=np.array([0, 0, -10.]),
                nd_per_contact=2,
                contact_detection_tolerance=1.0,
                is_quasi_dynamic=True,
                requires_grad=True,
                grad_from_active_constraints=grad_active_py)

            sim_params_cpp = cpp_params_from_py_params(sim_params)
            sim_params_cpp.gradient_from_active_constraints = grad_active_cpp

            q_sim_cpp = QuasistaticSimulatorCpp(
                model_directive_path=model_directive_path,
                robot_stiffness_str=robot_stiffness_dict,
                object_sdf_paths=object_sdf_dict,
                sim_params=sim_params_cpp)

            q_sim = QuasistaticSimulator(
                model_directive_path=model_directive_path,
                robot_stiffness_dict=robot_stiffness_dict,
                object_sdf_paths=object_sdf_dict,
                sim_params=sim_params,
                internal_vis=False)

            q0_dict = create_dict_keyed_by_model_instance_index(
                q_sim.get_plant(), q0_dict_str)

            q0_dict_cpp = create_dict_keyed_by_model_instance_index(
                q_sim_cpp.get_plant(), q0_dict_str)

            q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log = simulate(
                q_sim, q0_dict, h, T)

            q_dict_log_cpp, Dq_nextDq_log_cpp, Dq_nextDqa_cmd_log_cpp = simulate(
                q_sim_cpp, q0_dict_cpp, h, T)

            models_all = q_sim.models_all
            # match trajectories.
            for q_dict, q_dict_cpp in zip(q_dict_log, q_dict_log_cpp):
                for model in models_all:
                    self.assertTrue(np.allclose(q_dict[model], q_dict_cpp[model]))

            # match gradients along trajectories.
            # print('-------------------------------------------------')
            for i in range(len(Dq_nextDq_log)):
                Dq_next_Dq = Dq_nextDq_log[i]
                Dq_next_Dq_cpp = Dq_nextDq_log_cpp[i]
                Dq_nextDqa_cmd = Dq_nextDqa_cmd_log[i]
                Dq_nextDqa_cmd_cpp = Dq_nextDqa_cmd_log_cpp[i]
                # print(i, abs(Dq_next_Dq - Dq_next_Dq_cpp).max(),
                #       abs(Dq_nextDqa_cmd - Dq_nextDqa_cmd_cpp).max())

                self.assertTrue(np.allclose(Dq_next_Dq, Dq_next_Dq_cpp, atol=atol))
                self.assertTrue(np.allclose(Dq_nextDqa_cmd, Dq_nextDqa_cmd_cpp,
                                            atol=atol))
