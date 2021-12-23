import unittest

import numpy as np
from pydrake.autodiffutils import (initializeAutoDiff, AutoDiffXd,
                                   autoDiffToGradientMatrix)

from .normalization_derivatives import calc_normalization_derivatives


class TestNormalizationDerivatives(unittest.TestCase):
    def test_normalization_derivatives(self):
        q = np.array([1, 2, 3, 4], dtype=float)
        q *= 0.99 / np.linalg.norm(q)

        D = calc_normalization_derivatives(q)

        q_ad = initializeAutoDiff(q)
        q_bar_ad = q_ad / np.linalg.norm(q_ad)
        D_ad = autoDiffToGradientMatrix(q_bar_ad)

        self.assertTrue(np.allclose(D, D_ad))
