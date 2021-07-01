from unittest import TestCase
import numpy as np

from stopy.montecarlo import MonteCarlo


class TestMonteCarlo(TestCase):
    def test_mean(self):
        mc = MonteCarlo()
        self.assertEqual(mc.mean(lambda: 1.0), 1.0)
        data = np.linspace(1, 11, 19)
        self.assertEqual((mc.mean(lambda: data) - data).all(), 0.0)
        self.assertAlmostEqual(mc.mean(lambda: np.random.normal(loc=3.5),
                                       n_sim=10_000), 3.5, places=1)
        self.assertAlmostEqual(mc.mean(np.random.normal, param={"loc": -0.2},
                                       n_sim=100_000), -0.2, places=1)

    def test_mean_with_variance(self):
        mc = MonteCarlo()
        self.assertEqual(mc.mean_with_variance(lambda: 1.0)[1], 0.0)
        data = np.linspace(1, 11, 19)
        self.assertEqual((mc.mean_with_variance(lambda: data)[1]).all(), 0.0)
        self.assertAlmostEqual(mc.mean_with_variance(
            lambda: np.random.normal(loc=3.5), n_sim=10_000)[1], 1.0, places=1)
        self.assertAlmostEqual(mc.mean_with_variance(
            np.random.normal, {"scale": 0.2}, n_sim=100_000)[1], 0.04, places=1)

