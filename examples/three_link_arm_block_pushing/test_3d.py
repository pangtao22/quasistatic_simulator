import unittest
from examples.three_link_arm_block_pushing.run_3link_arm_pushing_3d import *


class Test3linkArmBoxPushing3D(unittest.TestCase):
    def test_3link_arm_box_pushing_3d(self):
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
            run_mbp_quasistatic_comparison(
                model_path_3d, q0_dict_str,
                is_visualizing=False,
                real_time_rate=0.0)

        (e_robot, e_vec_robot, t_e_robot,
         e_angle_box, e_vec_angle_box, t_angle_box,
         e_xyz_box, e_vec_xyz_box, t_xyz_box) = calc_integral_errors(
            q_robot_log_mbp, q_box_log_mbp, t_mbp,
            q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic)

        # Quasistatic vs MBP, robot.
        self.assertLessEqual(e_robot, 0.1)

        # Quasistatic vs MBP, object orientation, in terms of angles (rad).
        self.assertLessEqual(e_angle_box, 0.4)

        # Object position.
        self.assertLessEqual(e_xyz_box, 0.4)


if __name__ == '__main__':
    unittest.main()
