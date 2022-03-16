import os
import unittest

import numpy as np
from examples.setup_simulations import run_quasistatic_sim
from pydrake.all import PiecewisePolynomial
from qsim.simulator import (ForwardDynamicsMode, QuasistaticSimulator,
                            GradientMode)
from qsim.model_paths import models_dir
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim_old.problem_definition_graze import problem_definition
from qsim_old.simulator import QuasistaticSimulator as QsimOld


class TestBoxBallGraze(unittest.TestCase):
    def setUp(self):
        self.h = problem_definition['h']
        self.parser = QuasistaticParser(
            os.path.join(models_dir, 'q_sys', 'ball_grazing_2d.yml'))

    def test_old_vs_new(self):
        self.parser.set_sim_params(is_quasi_dynamic=True, h=self.h)

        # model names
        robot_name = 'ball'
        object_name = "box"
        np.testing.assert_allclose(
            self.parser.robot_stiffness_dict[robot_name],
            problem_definition['Kq_a'].diagonal())

        # trajectory and initial conditions
        nq_a = 2
        qa_knots = np.zeros((2, nq_a))
        qa_knots[0] = [0, 0.1]
        qa_knots[1] = [0.1, -0.1]

        n_steps = 10
        t_knots = [0, n_steps * self.h]
        qa_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)

        q_a_traj_dict_str = {robot_name: qa_traj}
        qu0 = np.array([0.])
        q0_dict_str = {object_name: qu0,
                       robot_name: qa_knots[0]}

        # Run sim with new simulator.
        loggers_dict_quasistatic_str, __ = run_quasistatic_sim(
            q_parser=self.parser,
            backend=QuasistaticSystemBackend.PYTHON,
            q_a_traj_dict_str=q_a_traj_dict_str,
            q0_dict_str=q0_dict_str,
            is_visualizing=False,
            real_time_rate=0)

        # initial condition in the form of [qu, qa], used by QsimOld.
        q0 = np.array([0, 0, 0.1])
        q = q0.copy()
        q_log = [q0.copy()]
        q_sim = QsimOld(problem_definition, is_quasi_dynamic=True)

        # Create a new qa_traj. This is necessary because run_quasistatic_sim
        # shifts qa_traj back by h...
        qa_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)

        for i in range(n_steps - 1):
            q_a_cmd = qa_traj.value((i + 1) * self.h).squeeze()
            dq_a, dq_u, lambda_n, lambda_f, result = q_sim.step_anitescu(
                q, q_a_cmd)

            # Update q.
            q += np.hstack([dq_u, dq_a])

            # logging.
            q_log.append(q.copy())

        q_log = np.array(q_log)

        # compare
        qu_old = q_log[:, 0]
        qa_old = q_log[:, 1:3]
        qu_new = loggers_dict_quasistatic_str[object_name].data()
        qa_new = loggers_dict_quasistatic_str[robot_name].data().T

        self.assertTrue(np.allclose(qu_new, qu_old))
        self.assertTrue(np.allclose(qa_old, qa_new))

    def test_log_dynamics_gradients(self):
        self.parser.set_sim_params(
            h=problem_definition['h'],
            gravity=np.array([0, 0, 0.]),
            nd_per_contact=2,
            contact_detection_tolerance=np.inf,
            is_quasi_dynamic=True,
            log_barrier_weight=100)

        q_sim = self.parser.make_simulator_py(internal_vis=False)
        q_sim_cpp = self.parser.make_simulator_cpp()
        q_sim_params = QuasistaticSimulator.copy_sim_params(
            self.parser.q_sim_params)

        model_u = q_sim.plant.GetModelInstanceByName("box")
        model_a = q_sim.plant.GetModelInstanceByName("ball")

        # Nominal state and action.
        q_a_cmd_dict = {model_a: np.array([0.1, -0.1])}
        q0_dict = {model_u: np.array([0.]),
                   model_a: np.array([0, 0.])}

        # C++ gradient
        q_sim_params.gradient_mode = GradientMode.kBOnly
        q_sim_params.forward_mode = ForwardDynamicsMode.kLogPyramidMp
        q_sim_cpp.update_mbp_positions(q0_dict)
        tau_ext_dict = q_sim_cpp.calc_tau_ext([])
        q_sim_cpp.step(q_a_cmd_dict=q_a_cmd_dict,
                       tau_ext_dict=tau_ext_dict,
                       sim_params=q_sim_params)
        B_cpp = q_sim_cpp.get_Dq_nextDqa_cmd()

        # Py gradient
        q_sim.update_mbp_positions(q0_dict)
        tau_ext_dict = q_sim.calc_tau_ext([])
        q_sim.step(q_a_cmd_dict=q_a_cmd_dict,
                   tau_ext_dict=tau_ext_dict,
                   sim_params=q_sim_params)
        B_py = q_sim.get_Dq_nextDqa_cmd()

        # Numerical gradient.
        q_sim_params.gradient_mode = GradientMode.kNone
        B_numerical = q_sim.calc_dfdu_numerical(q_dict=q0_dict,
                                                qa_cmd_dict=q_a_cmd_dict,
                                                du=1e-3,
                                                sim_params=q_sim_params)


        np.testing.assert_allclose(B_cpp, B_py)
        np.testing.assert_allclose(B_cpp, B_numerical, atol=1e-4)



