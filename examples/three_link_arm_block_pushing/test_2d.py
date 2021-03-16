import unittest
from .run_3link_arm_pushing_2d import *


class Test3linkArmBoxPushing2D(unittest.TestCase):
    def test_3link_arm_box_pushing_2d(self):
        """
        Another example of "unstable pushing". However, the difference
            between MBP and quasistatic is not as big as the 3D unstable
            pushing case, as motions in 2D are more confined.
        """
        (q_robot_log_mbp, q_box_log_mbp, t_mbp,
         q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic, _) = \
            run_comparison(box2d_big_sdf_path, q0_dict_str,
                           nd_per_contact=2,
                           is_visualizing=False, real_time_rate=0.0)

        (e_robot, e_vec_robot, t_e_robot,
         e_angle_box, e_vec_angle_box, t_angle_box,
         e_xyz_box, e_vec_xyz_box, t_xyz_box) = calc_integral_errors(
            q_robot_log_mbp, q_box_log_mbp, t_mbp,
            q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic)

        # Quasistatic vs MBP, robot.
        self.assertLessEqual(e_robot, 0.1)

        # Quasistatic vs MBP, object angle.
        self.assertLessEqual(e_angle_box, 0.1)

        # Quasistatic vs MBP, object position.
        self.assertLessEqual(e_xyz_box, 0.1)


if __name__ == '__main__':
    unittest.main()
