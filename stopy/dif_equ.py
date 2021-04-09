import numpy as np


class EulerScheme:
    def __init__(self, a_func=None, b_func=None, init_val=None):
        """Constructor for stochastic EulerScheme.
         It is used to solve character equations:
         dX_t = a(t, X_t)dt + b(t, X_t)dW_t
        =========================================================
        :param a_func: a function of two arguments, time and a spatial variable
        :param b_func: a function of two arguments, time and a spatial variable
        :param init_val: initial condition of the equation
        """
        self.a_func = a_func if a_func is not None else lambda t, x: 0
        self.b_func = b_func if b_func is not None else lambda t, x: 1
        self.init_val = 0.0 if init_val is None else init_val

    def _step(self, init, t_0=0.0, time=1.0, steps=10):
        """Internal function that computes the step of Euler's schema.
        :param init: numpy.array or float, initial condition
        :param t_0: float, beginning of time segment
        :param time: float, ending of time segment
        :param steps: int, number of time segment division
        :return: numpy.array or float
        """
        interval, _point = time - t_0, init
        size = np.array(_point).shape if isinstance(_point, np.ndarray) else 1
        for i in range(steps):
            arg_time = t_0 + (i * interval) / steps
            _point = _point\
                + self.a_func(arg_time, _point) * (interval / steps) \
                + self.b_func(arg_time, _point) * np.sqrt(interval / steps)\
                * np.random.normal(size=size)
        return _point

    def generate(self, t_0=0.0, dt=1.e-4, steps=1):
        """Method returning the generator of successive points from the
         trajectories of the X_t process.
        :param t_0: float, beginning of time
        :param dt: time "gain"
        :param steps: the number of steps in the schema that will be taken in
            each time increment but not returned. This means that the scheme is
            computed with time increments of dt / steps
        :return: generator generating successive elements from
            the trajectories of the process
        """
        _point, _time = self.init_val, t_0
        while True:
            yield _point
            _point = self._step(init=_point, t_0=_time,
                                time=_time + dt, steps=steps)
            _time = _time + dt
