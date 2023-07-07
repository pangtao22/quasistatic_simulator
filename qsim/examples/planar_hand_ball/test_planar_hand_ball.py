import os.path
import unittest
from typing import Union, Dict

import numpy as np
from pydrake.all import ModelInstanceIndex
from qsim.model_paths import models_dir
from qsim.parser import (
    QuasistaticParser,
    QuasistaticSimulator,
    QuasistaticSimulatorCpp,
)
from qsim.simulator import (
    GradientMode,
    QuasistaticSimParameters,
    ForwardDynamicsMode,
)
from qsim.utils import is_mosek_gurobi_available

q_model_path = os.path.join(models_dir, "q_sys", "planar_hand_ball.yml")


def simulate(
    sim: Union[QuasistaticSimulator, QuasistaticSimulatorCpp],
    q0_dict: Dict[ModelInstanceIndex, np.ndarray],
    T: int,
    sim_params: QuasistaticSimParameters,
):
    q_dict = {model: np.array(q_model) for model, q_model in q0_dict.items()}
    q_dict_log = [
        {model: np.array(q_model) for model, q_model in q_dict.items()}
    ]
    Dq_nextDq_log = []
    Dq_nextDqa_cmd_log = []
    sim.update_mbp_positions(q_dict)

    for _ in range(T):
        tau_ext_dict = sim.calc_tau_ext([])
        sim.step(q0_dict, tau_ext_dict, sim_params)
        q_dict_log.append(sim.get_mbp_positions())
        Dq_nextDq_log.append(sim.get_Dq_nextDq())
        Dq_nextDqa_cmd_log.append(sim.get_Dq_nextDqa_cmd())

    return q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log


def compare_q_dict_logs(
    test_case: unittest.TestCase,
    q_sim: QuasistaticSimulator,
    q_dict_log1,
    q_dict_log2,
    tol=1e-7,
):
    models_all = q_sim.models_all
    test_case.assertEqual(len(q_dict_log1), len(q_dict_log2))
    t = 0
    for q_dict, q_dict_cpp in zip(q_dict_log1, q_dict_log2):
        for model in models_all:
            err = np.linalg.norm(q_dict[model] - q_dict_cpp[model])
            test_case.assertLess(
                err,
                tol,
                f"Large error at t = {t}, "
                f"{q_dict[model]} vs {q_dict_cpp[model]}",
            )
        t += 1


class TestPlanarHandBall(unittest.TestCase):
    def setUp(self):
        self.h = 0.05
        # Num of time steps to simulate forward.
        self.T = int(round(2 / self.h))
        self.parser = QuasistaticParser(q_model_path)
        self.parser.set_sim_params(
            use_free_solvers=not is_mosek_gurobi_available(),
        )

        self.q_sim_py = self.parser.make_simulator_py(False)
        self.q_sim_cpp = self.parser.make_simulator_cpp()

        # model instance names.
        robot_l_name = "arm_left"
        robot_r_name = "arm_right"
        object_name = "sphere"

        name_to_model_dict = (
            self.q_sim_py.get_model_instance_name_to_index_map()
        )
        self.idx_l = name_to_model_dict[robot_l_name]
        self.idx_r = name_to_model_dict[robot_r_name]
        self.idx_o = name_to_model_dict[object_name]

        self.q0_dict = {
            self.idx_o: np.array([0, 0.5, 0]),
            self.idx_l: np.array([-np.pi / 4, -np.pi / 4]),
            self.idx_r: np.array([np.pi / 4, np.pi / 4]),
        }

    def test_cpp_vs_python(self):
        """
        Requires python bindings of QuiasistaticSimulator be on the PYTHONPATH.
        Test 3 cases for gradient computation:
        1. from all constraints, cpp vs. python.
        2. from active constraints, cpp vs. python.
        3. cpp from active constraints, python from all constraints.
        """
        atol = 1e-7
        sim_params = self.q_sim_py.get_sim_parmas_copy()
        sim_params.h = self.h
        sim_params.is_quasi_dynamic = True
        sim_params.gradient_mode = GradientMode.kAB

        q_dict_log_cpp, Dq_nextDq_log_cpp, Dq_nextDqa_cmd_log_cpp = simulate(
            self.q_sim_cpp, self.q0_dict, self.T, sim_params
        )

        q_dict_log, Dq_nextDq_log, Dq_nextDqa_cmd_log = simulate(
            self.q_sim_py, self.q0_dict, self.T, sim_params
        )

        # compare trajectories.
        compare_q_dict_logs(
            test_case=self,
            q_sim=self.q_sim_py,
            q_dict_log1=q_dict_log,
            q_dict_log2=q_dict_log_cpp,
        )

        # compare gradients along trajectories.
        for t in range(len(Dq_nextDq_log)):
            Dq_next_Dq = Dq_nextDq_log[t]
            Dq_next_Dq_cpp = Dq_nextDq_log_cpp[t]
            Dq_nextDqa_cmd = Dq_nextDqa_cmd_log[t]
            Dq_nextDqa_cmd_cpp = Dq_nextDqa_cmd_log_cpp[t]
            err_Dq_next_Dq = abs(Dq_next_Dq - Dq_next_Dq_cpp).max()
            err_Dq_nextDqa_cmd = abs(Dq_nextDqa_cmd - Dq_nextDqa_cmd_cpp).max()

            self.assertLess(err_Dq_next_Dq, atol, f"Large error at t = {t}")
            self.assertLess(
                err_Dq_nextDqa_cmd, atol, f"Large error at t = {t}"
            )

    def test_log_barrier(self):
        sim_params = self.q_sim_py.get_sim_parmas_copy()
        sim_params.h = self.h
        sim_params.is_quasi_dynamic = True
        sim_params.forward_mode = ForwardDynamicsMode.kLogPyramidCvx
        sim_params.gradient_mode = GradientMode.kNone
        sim_params.log_barrier_weight = 100

        q_dict_log_cvx, _, _ = simulate(
            self.q_sim_py, self.q0_dict, self.T, sim_params
        )

        sim_params.forward_mode = ForwardDynamicsMode.kLogPyramidMp
        q_dict_log_mp, _, _ = simulate(
            self.q_sim_py, self.q0_dict, self.T, sim_params
        )
        compare_q_dict_logs(
            test_case=self,
            q_sim=self.q_sim_py,
            q_dict_log1=q_dict_log_cvx,
            q_dict_log2=q_dict_log_mp,
            tol=1e-5,
        )

    def test_B(self):
        sim_params = self.q_sim_py.get_sim_parmas_copy()
        sim_params.h = self.h
        sim_params.is_quasi_dynamic = True
        sim_params.forward_mode = ForwardDynamicsMode.kQpMp
        sim_params.gradient_mode = GradientMode.kBOnly

        q_dict = {
            self.idx_o: [0, 0.314, 0],
            self.idx_l: [-0.775, -0.785],
            self.idx_r: [0.775, 0.785],
        }

        # Python gradient
        self.q_sim_py.update_mbp_positions(q_dict)
        tau_ext_dict = self.q_sim_py.calc_tau_ext([])
        self.q_sim_py.step(
            q_a_cmd_dict=q_dict,
            tau_ext_dict=tau_ext_dict,
            sim_params=sim_params,
        )
        dfdu_py = self.q_sim_py.get_Dq_nextDqa_cmd()

        # CPP gradient
        self.q_sim_cpp.update_mbp_positions(q_dict)
        self.q_sim_cpp.step(
            q_a_cmd_dict=q_dict,
            tau_ext_dict=tau_ext_dict,
            sim_params=sim_params,
        )
        dfdu_cpp = self.q_sim_cpp.get_Dq_nextDqa_cmd()

        # Numerical gradient
        dfdu_numerical = self.q_sim_py.calc_dfdu_numerical(
            q_dict=q_dict, qa_cmd_dict=q_dict, du=1e-4, sim_params=sim_params
        )

        np.testing.assert_allclose(dfdu_py, dfdu_cpp, atol=1e-8)
        np.testing.assert_allclose(dfdu_cpp, dfdu_numerical, atol=2e-3)
