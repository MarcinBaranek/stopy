from unittest import TestCase
import numpy as np
from stopy.processes import Wiener, Poisson, OrnsteinUhlenbeck


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

    def test_generate_on_grid(self):
        self.fail()

    def test_generate_array(self):
        self.fail()


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

    def test_generate_one_dim_jumps(self):
        self.fail()

    def test_generate_jumps(self):
        self.fail()

    def test_generate_on_grid(self):
        self.fail()


class TestOrnsteinUhlenbeck(TestCase):
    def test_generate(self):
        arguments = {"theta": 1.0, "mu": 0, "sigma": 1,
                     "start_point": np.zeros(shape=(3,)),
                     "time": 1, "x_axis": False, "division": 100}
        shape_div_100 = OrnsteinUhlenbeck.generate_array(**arguments)
        arguments["division"] = 200
        arguments["start_point"] = np.zeros(shape=(4,))
        shape_div_200 = OrnsteinUhlenbeck.generate_array(**arguments)
        arguments["division"] = 500
        arguments["time"] = 20
        arguments["start_point"] = np.zeros(shape=(5,))
        shape_time_20 = OrnsteinUhlenbeck.generate_array(**arguments)
        arguments["time"] = 8.5
        arguments["start_point"] = np.zeros(shape=(6,))
        shape_time_8coma5 = OrnsteinUhlenbeck.generate_array(**arguments)
        self.assertEqual(shape_div_100.shape, (100, 3))
        self.assertEqual(shape_div_200.shape, (200, 4))
        self.assertEqual(shape_time_20.shape, (10000, 5))
        self.assertEqual(shape_time_8coma5.shape, (4250, 6))

    def test_generate_on_grid(self):
        self.fail()

    def test_generate_array(self):
        self.fail()


class TestMemory(TestCase):
    def test_Wiener(self):
        self.fail()

    def test_Poisson(self):
        self.fail()

    def test_OrnsteinUhlenbeck(self):
        self.fail()


class TimeComputing(TestCase):
    def test_Wiener(self):
        self.fail()

    def test_Poisson(self):
        self.fail()

    def test_OrnsteinUhlenbeck(self):
        self.fail()
