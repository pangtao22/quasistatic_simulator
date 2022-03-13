import os
import unittest

import numpy as np
from examples.setup_simulations import run_quasistatic_sim
from pydrake.all import PiecewisePolynomial
from qsim.model_paths import models_dir
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim_old.problem_definition_graze import problem_definition
from qsim_old.simulator import QuasistaticSimulator as QsimOld


class TestBoxBallGrazeOldVsNew(unittest.TestCase):
    def test_sim_trajectory(self):
        h = problem_definition['h']
        parser = QuasistaticParser(
            os.path.join(models_dir, 'q_sys', 'ball_grazing_2d.yml'))
        parser.set_sim_params(is_quasi_dynamic=True, h=h)

        # model names
        robot_name = 'ball'
        object_name = "box"
        self.assertTrue(np.allclose(parser.robot_stiffness_dict[robot_name],
                                    problem_definition['Kq_a'].diagonal()))

        # trajectory and initial conditions
        nq_a = 2
        qa_knots = np.zeros((2, nq_a))
        qa_knots[0] = [0, 0.1]
        qa_knots[1] = [0.1, -0.1]

        n_steps = 10
        t_knots = [0, n_steps * h]
        qa_traj = PiecewisePolynomial.FirstOrderHold(t_knots, qa_knots.T)

        q_a_traj_dict_str = {robot_name: qa_traj}
        qu0 = np.array([0.])
        q0_dict_str = {object_name: qu0,
                       robot_name: qa_knots[0]}

        # Run sim with new simulator.
        loggers_dict_quasistatic_str, __ = run_quasistatic_sim(
            q_parser=parser,
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
            q_a_cmd = qa_traj.value((i + 1) * h).squeeze()
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
