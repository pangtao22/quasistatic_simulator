import os.path
import unittest
import sys
from typing import Union, Dict

import numpy as np
from examples.planar_hand_ball.run_planar_hand import q0_dict_str
from examples.setup_simulations import (
    create_dict_keyed_by_model_instance_index)
from pydrake.all import ModelInstanceIndex
from qsim.parser import (QuasistaticParser, QuasistaticSimulator,
                         QuasistaticSimulatorCpp)
from qsim.model_paths import models_dir
from qsim.sim_parameters import GradientMode

q_model_path = os.path.join(
    models_dir, 'q_sys', 'planar_hand_ball_vertical.yml')


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
        atol_list = [1e-7, 1e-7, 1e-4]

        for grad_active_py, grad_active_cpp, atol in zip(grad_active_py_list,
                                                         grad_active_cpp_list,
                                                         atol_list):
            parser = QuasistaticParser(q_model_path)

            # python sim
            parser.set_sim_params(
                is_quasi_dynamic=True, gradient_mode=GradientMode.kAB,
                grad_from_active_constraints=grad_active_py,
                gravity=[0, 0, -10.])

            q_sim = parser.make_simulator_py(internal_vis=False)

            # cpp sim
            parser.set_sim_params(grad_from_active_constraints=grad_active_cpp)
            q_sim_cpp = parser.make_simulator_cpp()

            q0_dict = create_dict_keyed_by_model_instance_index(
                q_sim.get_plant(), q0_dict_str)

            q0_dict_cpp = create_dict_keyed_by_model_instance_index(
                q_sim_cpp.get_plant(), q0_dict_str)

            # # -----------------------------------------------------------------
            # q_dict_log_list = []
            # Dq_nextDq_log_list = []
            # Dq_nextDqa_cmd_log_list = []
            # for _ in range(10):
            #     q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log = simulate(
            #         q_sim, q0_dict, h, T)
            #     q_dict_log_list.append(q_dict_log)
            #     Dq_nextDq_log_list.append(Dq_nextDq_log)
            #     Dq_nextDqa_cmd_log_list.append(Dq_nextDqa_cmd_log)
            #
            # # -----------------------------------------------------------------
            # q_dict_log_cpp_list = []
            # Dq_nextDq_log_cpp_list = []
            # Dq_nextDqa_cmd_log_cpp_list = []
            # for _ in range(10):
            #     q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log = simulate(
            #         q_sim_cpp, q0_dict, h, T)
            #     q_dict_log_cpp_list.append(q_dict_log)
            #     Dq_nextDq_log_cpp_list.append(Dq_nextDq_log)
            #     Dq_nextDqa_cmd_log_cpp_list.append(Dq_nextDqa_cmd_log)
            #
            # # -----------------------------------------------------------------

            q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log = simulate(
                q_sim, q0_dict, h, T)

            q_dict_log_cpp, Dq_nextDq_log_cpp, Dq_nextDqa_cmd_log_cpp = \
                simulate(q_sim_cpp, q0_dict_cpp, h, T)

            models_all = q_sim.models_all
            # match trajectories.
            for q_dict, q_dict_cpp in zip(q_dict_log, q_dict_log_cpp):
                for model in models_all:
                    self.assertTrue(np.allclose(q_dict[model],
                                                q_dict_cpp[model]))

            # match gradients along trajectories.
            print('------------------------------------------------------')
            for i in range(len(Dq_nextDq_log)):
                Dq_next_Dq = Dq_nextDq_log[i]
                Dq_next_Dq_cpp = Dq_nextDq_log_cpp[i]
                Dq_nextDqa_cmd = Dq_nextDqa_cmd_log[i]
                Dq_nextDqa_cmd_cpp = Dq_nextDqa_cmd_log_cpp[i]
                print(i, abs(Dq_next_Dq - Dq_next_Dq_cpp).max(),
                      abs(Dq_nextDqa_cmd - Dq_nextDqa_cmd_cpp).max())

                self.assertTrue(np.allclose(Dq_next_Dq, Dq_next_Dq_cpp,
                                            atol=atol))
                self.assertTrue(np.allclose(Dq_nextDqa_cmd, Dq_nextDqa_cmd_cpp,
                                            atol=atol))
