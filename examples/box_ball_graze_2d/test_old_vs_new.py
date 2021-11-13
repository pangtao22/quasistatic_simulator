import unittest
import os

import numpy as np
from pydrake.all import PiecewisePolynomial

from qsim_old.problem_definition_graze import problem_definition

from qsim_old.simulator import QuasistaticSimulator as QsimOld
from qsim.simulator import QuasistaticSimParameters

from examples.setup_simulation_diagram import run_quasistatic_sim
from examples.model_paths import models_dir
object_sdf_path = os.path.join(models_dir, "box_y.sdf")
model_directive_path = os.path.join(models_dir, "box_ball_graze_2d.yml")


class TestBoxBallGrazeOldVsNew(unittest.TestCase):
    def test_sim_trajectory(self):
        h = problem_definition['h']
        quasistatic_sim_params = QuasistaticSimParameters(
            gravity=np.array([0, 0, 0.]),
            nd_per_contact=2,
            contact_detection_tolerance=np.inf,
            is_quasi_dynamic=True,
            requires_grad=False)

        # robot
        Kp = problem_definition['Kq_a'].diagonal()
        robot_name = 'ball'
        robot_stiffness_dict = {robot_name: Kp}

        # object
        object_name = "box"
        object_sdf_dict = {object_name: object_sdf_path}

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
            model_directive_path=model_directive_path,
            object_sdf_paths=object_sdf_dict,
            q_a_traj_dict_str=q_a_traj_dict_str,
            q0_dict_str=q0_dict_str,
            robot_stiffness_dict={robot_name: Kp},
            h=h,
            sim_params=quasistatic_sim_params,
            is_visualizing=False,
            real_time_rate=0.)

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
