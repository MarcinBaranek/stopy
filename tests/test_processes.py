from unittest import TestCase
import numpy as np
from stopy.processes import Wiener, Poisson


class TestWiener(TestCase):
    def test_generate(self):
        shape_div_100 = Wiener.generate_array(np.zeros(shape=(3,)), time=1.0,
                                              division=100, x_axis=False)
        shape_div_200 = Wiener.generate_array(np.zeros(shape=(4,)), time=1.0,
                                              division=200, x_axis=False)
        shape_time_20 = Wiener.generate_array(np.zeros(shape=(5,)), time=20.0,
                                              x_axis=False)
        shape_time_8coma5 = Wiener.generate_array(np.zeros(shape=(6,)),
                                                  time=8.5, x_axis=False)
        self.assertEqual(shape_div_100.shape, (100, 3))
        self.assertEqual(shape_div_200.shape, (200, 4))
        self.assertEqual(shape_time_20.shape, (10000, 5))
        self.assertEqual(shape_time_8coma5.shape, (4250, 6))


class TestPoisson(TestCase):
    def test__generate_incrementally(self):
        args = {"lam": 1, "start_point": np.zeros(shape=(3,)),
                "time": 1, "division": 100}
        shape_div_100 = np.array(
            [element for element in Poisson._generate_incrementally(**args)])
        shape_div_200 = Wiener.generate_array(np.zeros(shape=(4,)), time=1.0,
                                              division=200, x_axis=False)
        shape_time_20 = Wiener.generate_array(np.zeros(shape=(5,)), time=20.0,
                                              x_axis=False)
        shape_time_8coma5 = Wiener.generate_array(np.zeros(shape=(6,)),
                                                  time=8.5, x_axis=False)
        self.assertEqual(shape_div_100.shape, (100, 3))
        self.assertEqual(shape_div_200.shape, (200, 4))
        self.assertEqual(shape_time_20.shape, (10000, 5))
        self.assertEqual(shape_time_8coma5.shape, (4250, 6))
