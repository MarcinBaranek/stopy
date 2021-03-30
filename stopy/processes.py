import numpy as np


class Wiener:
    """ Wiener process generator
    ============================================
    EXAMPLES:
        W = Wiener()

        # generate process to plot
        to_plot = W.generate_array(np.zeros(shape=(3,)), x_axis=True)
        plt.plot(to_plot[0], to_plot[1])
        plt.show()

        # generate process on grid
        grid = np.array([0.0, 1.9, 10.0, 30.0, 30.1, 32])
        plt.plot(grid, W.generate_on_grid(grid, start_point=np.zeros(shape=(4,))))
        plt.show()

        # do something until the process is in unit 5 dimensional ball
        trace = W.generate(np.zeros(shape=(5,)))
        point = trace.__next__()
        do_something(point)
        while (point ** 2).sum() < 1:
            point = trace.__next__()
            do_something(point)
    ============================================
    """
    @staticmethod
    def generate(start_point=np.zeros(shape=(1,)), time=1, division=500):
        """ function generating the process
        :param start_point: numpy.array with shape (n,), starting point of the process
        :param time: time interval, if time < 0, then function will be still generating process
        :param division: number of intermediate points in a unit time segment
        :return: generator generating successive realizations of the process
        """
        yield start_point
        if time < 0:
            while True:
                start_point = start_point + np.random.normal(loc=0.0, scale=1 / np.sqrt(division),
                                                             size=start_point.shape)
                yield start_point
        else:
            for _ in range(int(time * division)):
                start_point = start_point + np.random.normal(loc=0.0, scale=1 / np.sqrt(division),
                                                             size=start_point.shape)
                yield start_point

    @staticmethod
    def generate_on_grid(_grid, start_point=np.zeros(shape=(1,))):
        """ function generating the process on grid
        :param _grid: sorted list or array with grid
        :param start_point: numpy.array with shape (n,), starting point of the process
        :return: array with realizations of the Wiener process on grid
         """
        _array = [start_point]
        for i in range(1, len(_grid), 1):
            _array.append(_array[-1] + np.random.normal(loc=0.0, scale=1 / np.sqrt(_grid[i] - _grid[i - 1]),
                                                        size=start_point.shape))
        return np.array(_array)

    @staticmethod
    def generate_array(start_point=None, time=1, x_axis=False, division=500):
        """ function generating the process and return array of
        :param start_point: numpy.array with shape (n,), starting point of the process
        :param time: time interval, should be greater than 0
        :param x_axis: bool, if True return array with process and grid, else return only array with process
        :param division: number of intermediate points in a unit time segment
        :return: grid with array or only array with realizations of the Wiener process
        """
        start_point = start_point if start_point is not None else np.zeros(shape=(1,))
        _array = np.array([element for element in Wiener.generate(start_point, time=time, division=division)])
        return (np.linspace(0.0, time, int(time * division) + 1), _array) if x_axis else _array


class Poisson:
    """ Poisson process generator
    ============================================
    EXAMPLES:
        to_plot = Poisson.generate_jumps(dim=4, time=10)
        for arr in to_plot:
            plt.step(arr, np.arange(len(arr)))
        plt.show()

        grid = np.array([0.0, 1.9, 10.0, 30.0, 30.1, 32])
        for arr in Poisson.generate_on_grid(grid, start_point=np.zeros(shape=(5,))):
            plt.step(arr, np.arange(len(arr)))
        plt.show()
    ============================================
    """
    @staticmethod
    def generate_incrementally(lam, start_point=np.zeros(shape=(1,)), time=1, division=500):
        """ internal function generating the process
        :param lam: float, the intensity of the process
        :param start_point: numpy.array with shape (n,), starting point of the process
        :param time: time interval, if time < 0, then function will be still generating process
        :param division: number of intermediate points in a unit time segment
        :return: generator generating successive realizations of the process
        """
        yield start_point
        for _ in range(int(time * division)):
            start_point = start_point + np.random.poisson(lam/division, size=start_point.shape)
            yield start_point

    @staticmethod
    def generate_one_dim_jumps(lam=1, time=1):
        jumps = [0.0]
        while jumps[-1] < time:
            jumps.append(jumps[-1] + np.random.exponential(scale=1 / lam))
        return np.array(jumps)

    @staticmethod
    def generate_jumps(lam=1, dim=1, time=1):
        """ generates the process on the interval [0, time]
        :param lam: float, the intensity of the process
        :param dim: dimensionality of the process
        :param time: time interval
        :return: generator generating successive realizations of the process
        """
        jumps = []
        for d in range(dim):
            jumps.append(Poisson.generate_one_dim_jumps(lam=lam, time=time))
        return jumps

    @staticmethod
    def generate_on_grid(_grid, lam=1, start_point=np.zeros(shape=(1,))):
        """ function generating the process on grid
        :param _grid: sorted list or array with grid
        :param lam: float, the intensity of the process
        :param start_point: numpy.array with shape (n,), starting point of the process
        :return: array with realizations of the Poisson process on grid
         """
        _array = [start_point]
        for i in range(1, len(_grid), 1):
            _array.append(_array[-1] + np.random.poisson(lam=lam * (_grid[i] - _grid[i - 1]), size=start_point.shape))
        return np.array(_array).transpose()
