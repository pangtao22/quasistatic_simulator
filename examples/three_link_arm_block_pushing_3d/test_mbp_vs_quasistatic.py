import unittest
from .run_mbp_vs_quasistatic import *


class Test3linkArmBoxPushing(unittest.TestCase):
    def test_3link_arm_box_pushing(self):
        """
        This is an example of "unstable pushing", which manifests the
         difference contact models can make. MBP uses a penalty-based method
         to compute contact forces, where as QuasistaticSimulator uses
         Anitescu's convex model. As a result, although the two trajectories
         look similar, they are not identical (but are not too different
         either).

        The accuracy thershold are chosen based on a simulation run that
            looks reasonable.
        """
        (q_robot_log_mbp, q_box_log_mbp, t_mbp,
         q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic, _) = \
            run_comparison(is_visualizing=False, real_time_rate=0.0)

        # Set q_iiwa_traj to start at t=0.
        shift_q_traj_to_start_at_minus_h(q_robot_traj, 0)

        # Convert mbp knot points to a polynomial.
        qa_mbp_traj = PiecewisePolynomial.FirstOrderHold(
            t_mbp, q_robot_log_mbp.T)

        # Quasistatic vs MBP, robot.
        e_robot, e_vec_robot, t_e_robot = calc_error_integral(
            q_knots=q_robot_log_quasistatic,
            t=t_quasistatic,
            q_gt_traj=qa_mbp_traj)
        self.assertLessEqual(e_robot, 0.1)

        # Quasistatic vs MBP, object orientation, in terms of angles (rad).
        quaternion_box_mbp_traj = \
            convert_quaternion_array_to_eigen_quaternion_traj(
                q_box_log_mbp[:, :4], t_mbp)

        e_angle_box, e_vec_angle_box, t_angle_box = \
            calc_quaternion_error_integral(
                q_list=q_box_log_quasistatic[:, :4],
                t=t_quasistatic,
                q_traj=quaternion_box_mbp_traj)
        self.assertLessEqual(e_angle_box, 0.4)

        # Object position.
        xyz_box_mbp_traj = PiecewisePolynomial.FirstOrderHold(
            t_mbp, q_box_log_mbp[:, 4:].T)
        e_xyz_box, e_vec_xyz_box, t_xyz_box = calc_error_integral(
            q_knots=q_box_log_quasistatic[:, 4:],
            t=t_quasistatic,
            q_gt_traj=xyz_box_mbp_traj)
        self.assertLessEqual(e_xyz_box, 0.15)


if __name__ == '__main__':
    unittest.main()
