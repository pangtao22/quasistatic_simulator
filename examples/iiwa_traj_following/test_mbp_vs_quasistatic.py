import unittest
from .run_mbp_vs_quasistatic import *


class TestIiwaTrajectoryFollowing(unittest.TestCase):
    def test_traj_following_accuracy(self):
        """
        For reasons I don't yet understand, the integral error between
         MBP-simulated trajectories and the commanded trajectories converge
         to a non-zero value, even with long trajectory duration (100s) and
         small MBP time step (1e-5).

        On the otherhand, integral error of quasistatic-simulated
         trajectories approaches 0 as h_quasistatic decreases.
        """
        q_iiwa_log_mbp, t_mbp, q_iiwa_log_quasistatic, t_quasistatic = \
            run_comparison(is_visualizing=False, real_time_rate=0.0)

        # convert q_gt_knots to a piecewise polynomial.
        shift_q_traj_to_start_at_minus_h(q_iiwa_traj, 0)

        # MBP vs commanded.
        e2, e_vec2, t_e2 = calc_error_integral(
            q_knots=q_iiwa_log_mbp,
            t=t_mbp,
            q_gt_traj=q_iiwa_traj)
        self.assertLessEqual(e2, 0.08,
                             "Too much deviation from MBP-simulated "
                             "trajectory and commanded trajectory.")

        # Quasistatic vs commanded.
        e3, e_vec3, t_e3 = calc_error_integral(
            q_knots=q_iiwa_log_quasistatic,
            t=t_quasistatic,
            q_gt_traj=q_iiwa_traj)
        self.assertLessEqual(e3, 0.002,
                             "Too much deviation from Quasistatic-simulated "
                             "trajectory and commanded trajectory.")


if __name__ == '__main__':
    unittest.main()
