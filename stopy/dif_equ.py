import numpy as np
from scipy.stats import chisquare


class ItoProcess:
    """
    # ToDO write the documentation
    """
    def __init__(self, a_func=None, b_func=None, init_val=None):
        """
        # ToDO write the documentation
        :param a_func:
        :param b_func:
        :param init_val:
        """
        self.a_func = a_func if a_func is not None else lambda t, x: 0
        self.b_func = b_func if b_func is not None else lambda t, x: 1
        self.init_val = 0.0 if init_val is None else init_val

    def __eq__(self, other):
        """
        # ToDO write the documentation
        :param other:
        :return:
        """
        a_check = self.a_func.__code__.co_code == other.a_func.__code__.co_code
        b_check = self.b_func.__code__.co_code == other.b_func.__code__.co_code
        init_check = self.init_val == other.init_val
        return a_check and b_check and init_check

    def __str__(self):
        return f"Base Ito Process with initial value {self.init_val}\n" \
               f"{self.__repr__()}"

    def __add__(self, other):
        """
        # ToDO write the documentation
        :param other:
        :return:
        """
        return ItoProcess(lambda t, x: self.a_func(t, x) + other.a_func(t, x),
                          lambda t, x: self.b_func(t, x) + other.b_func(t, x),
                          self.init_val + other.init_val)

    def fit_test(self, data, t_arr=None, df=None, steps=10):
        """The method checks how well the process describes the data.
        =========================================================
        Statistical test:
        H_0: the data comes from the distribution described by the process
        H_1: The data is not from the distribution described by the process

        if p value < significance level then we reject the null hypothesis

        References:
        [1] Bak, J. (1998), Nonparametric methods in finance, Master’s thesis,
         Department of Mathematical Modelling, Technical University of Denmark,
         Lyngby. IMM-EKS-1998-34.
        =========================================================
        :param data: data array
        :param t_arr: time array, default is [0, 1, 2, ..., len_data]
        :param df: int, degrees of freedom, otherwise the number of simulations
            default is equal to int((len_data - 6) / 5)
        :param steps: int, the number of steps on which the Euler scheme is
            based in each simulation, default is 10
        :return: float, p value
        """
        # data preparation
        schema, len_data = EulerScheme(self.a_func, self.b_func), len(data)
        df = int((len_data - 6) / 5) if df is None else df
        expected = (len_data - 1) / (df + 1)
        t_arr = np.arange(1, len_data + 1) if t_arr is None else t_arr
        r_arr = np.ones(shape=(len_data - 1))

        # computing simulations
        for i in range(len_data - 1):
            for _ in range(df):
                r_arr[i] += int(
                    schema.step(data[i], t_arr[i], t_arr[i + 1], steps)
                    <= data[i + 1])

        # preparation for the test
        omega_arr = np.array([
            np.array(list(map(lambda x: 1 if x == i else 0, r_arr))).sum()
            for i in range(1, df + 1)])
        return chisquare(omega_arr, expected)[1]


class EulerScheme(ItoProcess):
    """Euler scheme class for solving stochastic differential equations
    # ToDO write the documentation
    """
    def __init__(self, a_func=None, b_func=None, init_val=None):
        """Constructor for stochastic EulerScheme.
         It is used to solve character equations:
         dX_t = a(t, X_t)dt + b(t, X_t)dW_t
        =========================================================
        :param a_func: a function of two arguments, time and a spatial variable
        :param b_func: a function of two arguments, time and a spatial variable
        :param init_val: initial condition of the equation
        """
        super().__init__(a_func, b_func, init_val)

    def step(self, init, t_0=0.0, time=1.0, steps=10):
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
            _point = self.step(init=_point, t_0=_time,
                               time=_time + dt, steps=steps)
            _time = _time + dt
