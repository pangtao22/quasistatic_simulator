import os.path
import unittest
from typing import Union, Dict

import numpy as np
from examples.setup_simulations import (
    create_dict_keyed_by_model_instance_index)
from pydrake.all import ModelInstanceIndex
from qsim.parser import (QuasistaticParser, QuasistaticSimulator,
                         QuasistaticSimulatorCpp)
from qsim.model_paths import models_dir
from qsim.simulator import (GradientMode, QuasistaticSimParameters,
                            ForwardDynamicsMode)

q_model_path = os.path.join(
    models_dir, 'q_sys', 'planar_hand_ball.yml')


def simulate(sim: Union[QuasistaticSimulator, QuasistaticSimulatorCpp],
             q0_dict: Dict[ModelInstanceIndex, np.ndarray], T: int):
    q_dict = {model: np.array(q_model)
              for model, q_model in q0_dict.items()}
    q_dict_log = [{model: np.array(q_model)
                   for model, q_model in q_dict.items()}]
    Dq_nextDq_log = []
    Dq_nextDqa_cmd_log = []
    sim.update_mbp_positions(q_dict)

    for _ in range(T):
        tau_ext_dict = sim.calc_tau_ext([])
        sim.step_default(q0_dict, tau_ext_dict)
        q_dict_log.append(sim.get_mbp_positions())
        Dq_nextDq_log.append(sim.get_Dq_nextDq())
        Dq_nextDqa_cmd_log.append(sim.get_Dq_nextDqa_cmd())

    return q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log


def compare_q_dict_logs(test_case: unittest.TestCase,
                        q_sim: QuasistaticSimulator,
                        q_dict_log1, q_dict_log2,
                        tol=1e-7):
    models_all = q_sim.models_all
    test_case.assertEqual(len(q_dict_log1), len(q_dict_log2))
    t = 0
    for q_dict, q_dict_cpp in zip(q_dict_log1, q_dict_log2):
        for model in models_all:
            err = np.linalg.norm(q_dict[model] - q_dict_cpp[model])
            test_case.assertLess(err, tol,  f"Large error at t = {t}, "
                                 f"{q_dict[model]} vs {q_dict_cpp[model]}")
        t += 1


class TestPlanarHandBall(unittest.TestCase):
    def setUp(self):
        self.h = 0.05
        # Num of time steps to simulate forward.
        self.T = int(round(2 / self.h))
        self.parser = QuasistaticParser(q_model_path)

        # model instance names.
        robot_l_name = "arm_left"
        robot_r_name = "arm_right"
        object_name = "sphere"

        self.q0_dict_str = {object_name: np.array([0, 0.5, 0]),
                       robot_l_name: np.array([-np.pi / 4, -np.pi / 4]),
                       robot_r_name: np.array([np.pi / 4, np.pi / 4])}

    def test_cpp_vs_python(self):
        """
        Requires python bindings of QuiasistaticSimulator be on the PYTHONPATH.
        Test 3 cases for gradient computation:
        1. from all constraints, cpp vs. python.
        2. from active constraints, cpp vs. python.
        3. cpp from active constraints, python from all constraints.
        """
        atol = 1e-7

        # python sim
        self.parser.set_sim_params(
            h=self.h,
            is_quasi_dynamic=True, gradient_mode=GradientMode.kAB,
            gravity=[0, 0, -10.])

        q_sim = self.parser.make_simulator_py(internal_vis=False)
        q_sim_cpp = self.parser.make_simulator_cpp()

        q0_dict = create_dict_keyed_by_model_instance_index(
            q_sim.get_plant(), self.q0_dict_str)

        q0_dict_cpp = create_dict_keyed_by_model_instance_index(
            q_sim_cpp.get_plant(), self.q0_dict_str)

        # # ----------------------------------------------------------------
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
        # # ----------------------------------------------------------------
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
        # # ----------------------------------------------------------------

        q_dict_log_cpp, Dq_nextDq_log_cpp, Dq_nextDqa_cmd_log_cpp = \
            simulate(q_sim_cpp, q0_dict_cpp, self.T)

        q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log = simulate(q_sim, q0_dict,
                                                                 self.T)

        # compare trajectories.
        compare_q_dict_logs(
            test_case=self,
            q_sim=q_sim,
            q_dict_log1=q_dict_log,
            q_dict_log2=q_dict_log_cpp)

        # compare gradients along trajectories.
        print('-------------------------------------------------------')
        for t in range(len(Dq_nextDq_log)):
            Dq_next_Dq = Dq_nextDq_log[t]
            Dq_next_Dq_cpp = Dq_nextDq_log_cpp[t]
            Dq_nextDqa_cmd = Dq_nextDqa_cmd_log[t]
            Dq_nextDqa_cmd_cpp = Dq_nextDqa_cmd_log_cpp[t]
            err_Dq_next_Dq = abs(Dq_next_Dq - Dq_next_Dq_cpp).max()
            err_Dq_nextDqa_cmd = abs(
                Dq_nextDqa_cmd - Dq_nextDqa_cmd_cpp).max()

            self.assertLess(err_Dq_next_Dq, atol,
                            f"Large error at t = {t}")
            self.assertLess(err_Dq_nextDqa_cmd, atol,
                            f"Large error at t = {t}")

    def test_cvx_vs_mp(self):
        self.parser.set_sim_params(
            h=self.h,
            is_quasi_dynamic=True, gradient_mode=GradientMode.kNone,
            forward_mode=ForwardDynamicsMode.kLogPyramidCvx,
            log_barrier_weight=100,
            gravity=[0, 0, -10.])
        q_sim = self.parser.make_simulator_py(False)
        q0_dict = create_dict_keyed_by_model_instance_index(
            q_sim.get_plant(), self.q0_dict_str)

        q_dict_log_cvx, _, _ = simulate(q_sim, q0_dict, self.T)

        self.parser.set_sim_params(
            forward_mode=ForwardDynamicsMode.kLogPyramidMp)
        q_sim = self.parser.make_simulator_py(False)
        q_dict_log_mp, _, _ = simulate(q_sim, q0_dict, self.T)

        compare_q_dict_logs(
            test_case=self,
            q_sim=q_sim,
            q_dict_log1=q_dict_log_cvx,
            q_dict_log2=q_dict_log_mp,
            tol=1e-5)
