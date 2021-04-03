import numpy as np


class EulerScheme:
    def __init__(self, a_func, b_func, init_val=0.0):
        self.a_func = a_func
        self.b_func = b_func
        self.init_val = init_val

    def _update_step(self, time, point, step):
        """ Compute one step in evaluate_path:
        a(t, X_t)dt + b(t, X_t)dW_t
        :param time: float, time
        :param point: numpy.array, current point
        :param step: float, it is dt
        :return: numpy.array with the same shape as return a_func and b_func
        """
        return self.a_func(time, point) * step + self.b_func(time, point) * np.random.normal() / np.sqrt(step)

    def evaluate_path(self, time=1.0, division=100):
        """ returns the generator that generates successive trajectory points
        :param time: float, end of time interval, if time < 0, then generate infinity trajectory
        :param division: int, number of intermediate points in the unit interval
        :return: generator
        """
        yield self.init_val
        step, _time, _point = 1.0 / division, 0.0, self.init_val
        if time < 0:
            while True:
                _time, _point = _time + step, self._update_step(time, _point, step)
                yield _point
        else:
            for _ in range(int(time / division)):
                _time, _point = _time + step, self._update_step(time, _point, step)
                yield _point
