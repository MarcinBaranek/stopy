import numpy as np
from scipy.stats import chisquare
from stopy.montecarlo import MonteCarlo as Mc


class EulerScheme:
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
        self.a_func = a_func if a_func is not None else lambda t, x: 0
        self.b_func = b_func if b_func is not None else lambda t, x: 1
        self.init_val = 0.0 if init_val is None else init_val

    def step(self, point=0.0, time=1.0, dt=1.e-4):
        """Internal function that computes the step of Euler's schema.
        :param point: numpy.array or float, initial condition
        :param time: float, beginning of time segment
        :param dt: float, time "gain"
        :return: numpy.array or float
        """
        # ToDo fix this below
        try:
            size = point.shape[0]
        except Exception:
            size = 1
        return point + self.a_func(time, point) * dt \
            + self.b_func(time, point) * np.sqrt(dt) \
            * np.random.normal(size=size)

    def generate(self, point=None, t_0=0.0, dt=1.e-4, end=None, grid=False):
        """Method returning the generator of successive points from the
         trajectories of the X_t process.
        :param point: float, beginning of time
        :param t_0: float, beginning of time
        :param dt: time "gain"
        :param end: ToDo write
        :param grid: ToDo write
        :return: generator generating successive elements from
            the trajectories of the process
        """
        point, time = self.init_val if point is None else point, t_0
        if end is not None:
            while time <= end:
                yield (point, time) if grid else point
                point = self.step(point, time, dt)
                time += dt
        else:
            while True:
                yield (point, time) if grid else point
                point = self.step(point, time, dt)
                time += dt


class ItoProcess(EulerScheme):
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
        super().__init__(a_func, b_func, init_val)

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

    def fit_test(self, data, t_arr=None, df=None):
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
        :param t_arr: time array, default is (1.e-4) * [0, 1, 2, ..., len_data]
        :param df: int, degrees of freedom, otherwise the number of simulations
            default is equal to int((len_data - 6) / 5)
        :return: float, p value
        """
        # data preparation
        len_data = len(data)
        df = int((len_data - 6) / 5) if df is None else df
        expected = (len_data - 1) / (df + 1)
        t_arr = np.arange(len_data) * 1.e-4 if t_arr is None else t_arr
        r_arr = np.ones(shape=(len_data - 1))

        # computing simulations
        for i in range(len_data - 1):
            for _ in range(df):
                # ToDO implement schema dependent of K
                r_arr[i] += int(
                    self.step(data[i], t_arr[i], t_arr[i + 1] - t_arr[i])
                    <= data[i + 1])

        # preparation for the test
        omega_arr = [sum(map(lambda x: 1 if x == i else 0, r_arr))
                     for i in range(1, df + 1)]
        return chisquare(omega_arr, expected)[1]
