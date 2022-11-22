import unittest

from examples.iiwa_block_stacking.run_mbp_vs_quasistatic import *
from examples.setup_simulations import compare_q_sim_cpp_vs_py


class TestIiwaBlockStacking(unittest.TestCase):
    def setUp(self) -> None:
        self.q_parser = QuasistaticParser(q_model_path)
        self.h_quasistatic = 0.1

    def test_cpp_vs_python(self):
        self.q_parser.set_sim_params(h=self.h_quasistatic)
        compare_q_sim_cpp_vs_py(
            test_case=self,
            q_parser=self.q_parser,
            q_a_traj_dict_str=q_a_traj_dict_str,
            q0_dict_str=q0_dict_str,
            atol=1e-4,
        )

    def test_mbp_vs_quasistatic(self):
        """
        This test compares MBP and quasistatic simulations. Some observations:
        - For IIWA, Quasistaitc is better at tracking trajectory,
            probably due to the absence of inertia. This is consistent with
            the test in iiwa_traj_tracking.

            Max joint angle error norm between q_cmd and q_quasistatic is
            0.006, mostly due to holding the red box. In constrast, max error
            between q_quasistatic and q_mbp is 0.06, and the peaks are
            coincident with accelerating/decelerating the box.

            Moreover, quasistatic tracking gets better as h_quasistatic
            decreases.

            Increasing the duration of the entire trajectory does reduce by
            a constant factor the error between q_cmd and q_mbp, but the
            error integral appears to remain constant, as the smaller error is
            integrated over a longer period of time.

        - Red box angle error is the largest when it is rotated, and peaks at
            the beginning and end of the rotation, possibly due to
            acceleration and deceleration.

        Accuracy thresholds are chosen based on a "visually reasonable" runs of
        both simulations.
        """
        (
            loggers_dict_mbp_str,
            loggers_dict_quasistatic_str,
            plant,
        ) = run_comparison(
            h_mbp=1e-3, h_quasistatic=self.h_quasistatic, is_visualizing=False
        )

        error_dict = compare_all_models(
            plant, loggers_dict_mbp_str, loggers_dict_quasistatic_str
        )

        for model_name, error in error_dict.items():
            if model_name == "box0":
                self.assertLessEqual(error, 0.2)
            elif model_name == iiwa_name:
                self.assertLessEqual(error, 0.52)
            else:
                self.assertLessEqual(error, 0.05)


if __name__ == "__main__":
    unittest.main()
