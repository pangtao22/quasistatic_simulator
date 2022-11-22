import unittest
from .run_iiwa_traj_following import *
from examples.setup_simulations import compare_q_sim_cpp_vs_py


class TestIiwaTrajectoryFollowing(unittest.TestCase):
    def test_cpp_vs_python(self):
        q_parser = QuasistaticParser(q_model_path)
        q_parser.set_sim_params(h=0.1)
        compare_q_sim_cpp_vs_py(
            test_case=self,
            q_parser=q_parser,
            q_a_traj_dict_str={robot_name: q_iiwa_traj},
            q0_dict_str=q0_dict_str,
            atol=1e-8,
        )

    def test_quasistatic_vs_mbp(self):
        """
        The integral error between MBP-simulated trajectories and the
         commanded trajectories converge to a non-zero value, even with long
         trajectory duration (100s) and small MBP time step (1e-5). Possible
         explanation: a second order system tracking a ramp input has non-zero
         steady-state error.

        On the other hand, integral error of quasistatic-simulated
         trajectories approaches 0 as h_quasistatic decreases.
        """
        (
            q_iiwa_log_mbp,
            t_mbp,
            q_iiwa_log_quasistatic,
            t_quasistatic,
        ) = run_comparison(is_visualizing=False, real_time_rate=0.0)

        # Set q_iiwa_traj to start at t=0.
        shift_q_traj_to_start_at_minus_h(q_iiwa_traj, 0)

        # MBP vs commanded.
        e2, e_vec2, t_e2 = calc_error_integral(
            q_knots=q_iiwa_log_mbp, t=t_mbp, q_gt_traj=q_iiwa_traj
        )
        self.assertLessEqual(
            e2,
            0.08,
            "Too much deviation from MBP-simulated "
            "trajectory and commanded trajectory.",
        )

        # Quasistatic vs commanded.
        e3, e_vec3, t_e3 = calc_error_integral(
            q_knots=q_iiwa_log_quasistatic,
            t=t_quasistatic,
            q_gt_traj=q_iiwa_traj,
        )
        self.assertLessEqual(
            e3,
            0.002,
            "Too much deviation from Quasistatic-simulated "
            "trajectory and commanded trajectory.",
        )


if __name__ == "__main__":
    unittest.main()
