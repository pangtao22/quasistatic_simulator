import unittest
from .run_iiwa_external_loading import *


class TestIiwaExternalLoading(unittest.TestCase):
    def test_mbp_vs_quaistatic(self):
        loggers_dict_quasistatic_str, loggers_dict_mbp_str = run_comparison(
            is_visualizing=False, real_time_rate=0)

        nq = 7
        iiwa_log_mbp = loggers_dict_mbp_str[iiwa_name]
        q_iiwa_mbp = iiwa_log_mbp.data().T[:, :nq]

        iiwa_log_qs = loggers_dict_quasistatic_str[iiwa_name]
        q_iiwa_qs = iiwa_log_qs.data().T[:, :nq]

        self.assertLessEqual(np.linalg.norm(q_iiwa_qs[-1] - q_iiwa_mbp[-1]),
                             1e-6)


if __name__ == '__main__':
    unittest.main()
