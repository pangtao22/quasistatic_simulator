import unittest
from .run_3link_arm_pushing_2d import *


class Test3linkArmBoxPushing2D(unittest.TestCase):
    def setUp(self):
        self.quasistatic_sim_params = QuasistaticSimParameters(
            gravity=gravity,
            nd_per_contact=2,
            contact_detection_tolerance=np.inf)

    def test_python_vs_cpp(self):
        loggers_dict_quasistatic_str_cpp, q_sys_cpp = run_quasistatic_sim(
            model_directive_path=model_directive_path,
            object_sdf_paths={box_name: box2d_big_sdf_path},
            q_a_traj_dict_str={robot_name: q_robot_traj},
            q0_dict_str=q0_dict_str,
            robot_stiffness_dict=robot_stiffness_dict,
            h=h_quasistatic,
            sim_params=self.quasistatic_sim_params,
            is_visualizing=False, real_time_rate=0.,
            backend="cpp")

        loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
            model_directive_path=model_directive_path,
            object_sdf_paths={box_name: box2d_big_sdf_path},
            q_a_traj_dict_str={robot_name: q_robot_traj},
            q0_dict_str=q0_dict_str,
            robot_stiffness_dict=robot_stiffness_dict,
            h=h_quasistatic,
            sim_params=self.quasistatic_sim_params,
            is_visualizing=False, real_time_rate=0.,
            backend="python")

        for name in loggers_dict_quasistatic_str_cpp.keys():
            q_log_cpp = loggers_dict_quasistatic_str_cpp[name].data()
            q_log = loggers_dict_quasistatic_str[name].data()

            self.assertEqual(q_log.shape, q_log_cpp.shape)
            self.assertTrue(np.allclose(q_log, q_log_cpp))

    def test_mbp_vs_quasistatic(self):
        """
        Another example of "unstable pushing". However, the difference
            between MBP and quasistatic is not as big as the 3D unstable
            pushing case, as motions in 2D are more confined.
        """

        (q_robot_log_mbp, q_box_log_mbp, t_mbp,
         q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic, _) = \
            run_mbp_quasistatic_comparison(
                box2d_big_sdf_path, q0_dict_str,
                quasistatic_sim_params=self.quasistatic_sim_params,
                is_visualizing=False, real_time_rate=0.0)

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
