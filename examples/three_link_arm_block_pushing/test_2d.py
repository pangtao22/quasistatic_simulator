import unittest
from .run_3link_arm_pushing_2d import *
from examples.setup_simulations import compare_q_sim_cpp_vs_py


class Test3linkArmBoxPushing2D(unittest.TestCase):
    def setUp(self):
        self.q_parser = QuasistaticParser(
            os.path.join(models_dir, q_model_path_2d))
        self.q_parser.set_sim_params(h=h_quasistatic)

    def test_python_vs_cpp(self):
        compare_q_sim_cpp_vs_py(test_case=self, q_parser=self.q_parser,
                                q_a_traj_dict_str={robot_name: q_robot_traj},
                                q0_dict_str=q0_dict_str, atol=1e-6)

    def test_mbp_vs_quasistatic(self):
        """
        Another example of "unstable pushing". However, the difference
            between MBP and quasistatic is not as big as the 3D unstable
            pushing case, as motions in 2D are more confined.
        """

        (q_robot_log_mbp, q_box_log_mbp, t_mbp,
         q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic,
         q_sys) = run_mbp_quasistatic_comparison(
            q_model_path_2d, q0_dict_str,
            is_visualizing=False,
            real_time_rate=0.0)

        (e_robot, e_vec_robot, t_e_robot,
         e_angle_box, e_vec_angle_box, t_angle_box,
         e_xyz_box, e_vec_xyz_box, t_xyz_box) = calc_integral_errors(
            q_robot_log_mbp, q_box_log_mbp, t_mbp,
            q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic)

        # Quasistatic vs MBP, robot.
        self.assertLessEqual(e_robot, 0.1)

        # Quasistatic vs MBP, object angle.
        self.assertLessEqual(e_angle_box, 0.15)

        # Quasistatic vs MBP, object position.
        self.assertLessEqual(e_xyz_box, 0.1)


if __name__ == '__main__':
    unittest.main()
