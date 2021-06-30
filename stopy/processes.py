import numpy as np


class Wiener:
    """Wiener process generator
    ============================================
    EXAMPLES:
        # generate process to plot
        to_plot = Wiener.generate_array(np.zeros(shape=(3,)), x_axis=True)
        plt.plot(to_plot[0], to_plot[1])
        plt.show()

        # generate process on grid
        grid = np.array([0.0, 1.9, 10.0, 30.0, 30.1, 32])
        plt.plot(grid,
            Wiener.generate_on_grid(grid, start_point=np.zeros(shape=(4,))))
        plt.show()

        # do something until the process is in unit 5 dimensional ball
        trace = Wiener.generate(np.zeros(shape=(5,)))
        point = trace.__next__()
        do_something(point)
        while (point ** 2).sum() < 1:
            point = trace.__next__()
            do_something(point)
    ============================================
    """
    def __init__(self, *args, **kwargs):
        """Empty init, because it is fully static class
        """
        pass

    @staticmethod
    def generate(start_point=None, time=1.0, division=500):
        """Function generating the process
        :param start_point: numpy.array with shape (n,),
            initial point of the process
        :param time: time interval, if time < 0,
            then function will be still generating process
        :param division: number of intermediate points in a unit time segment
        :return: generator generating successive realizations of the process
        """
        _point = np.zeros(shape=(1,)) if start_point is None else start_point
        dim = _point.shape
        if time < 0:
            while True:
                yield _point
                _point = _point + np.random.normal(size=dim) / np.sqrt(division)
        else:
            for _ in range(int(time * division)):
                yield _point
                _point = _point + np.random.normal(size=dim) / np.sqrt(division)

    @staticmethod
    def generate_on_grid(_grid, start_point=None):
        """Function generating the process on grid
        :param _grid: sorted list or array with grid
        :param start_point: numpy.array with shape (n,),
            starting point of the process
        :return: array with realizations of the Wiener process on grid
         """
        _arr = [np.zeros(shape=(1,))] if start_point is None else [start_point]
        size = _arr[0].shape
        for i in range(1, len(_grid), 1):
            _arr.append(_arr[-1] + np.random.normal(size)
                        / np.sqrt(_grid[i] - _grid[i - 1]))
        return np.array(_arr)

    @staticmethod
    def generate_array(start_point=None, time=1.0, x_axis=False, division=500):
        """Function generating the process and return array of
        :param start_point: numpy.array with shape (n,),
            starting point of the process
        :param time: time interval, should be greater than 0
        :param x_axis: bool, if True return array with process and grid,
            else return only array with process
        :param division: number of intermediate points in a unit time segment
        :return: grid with array or only array with realizations of the process
        """
        _point = np.zeros(shape=(1,)) if start_point is None else start_point
        _arr = np.array(
            [element for element in Wiener.generate(start_point=_point,
                                                    time=time,
                                                    division=division)])
        return (np.linspace(0.0, time, int(time * division)), _arr)\
            if x_axis else _arr


class Poisson:
    """Poisson process generator
    ============================================
    EXAMPLES:
        to_plot = Poisson.generate_jumps(dim=4, time=10)
        for arr in to_plot:
            plt.step(arr, np.arange(len(arr)))
        plt.show()

        grid = np.array([0.0, 1.9, 10.0, 30.0, 30.1, 32])
        for arr in Poisson.generate_on_grid(grid,
                                            start_point=np.zeros(shape=(5,))):
            plt.step(arr, np.arange(len(arr)))
        plt.show()
    ============================================
    """
    def __init__(self, *args, **kwargs):
        """Empty init, because it is fully static class
        """
        pass

    @staticmethod
    def _generate_incrementally(lam, start_point=None, time=1, division=500):
        """Internal function generating the process
        :param lam: float, the intensity of the process
        :param start_point: numpy.array with shape (n,),
            starting point of the process
        :param time: time interval, if time < 0,
            then function will be still generating process
        :param division: number of intermediate points in a unit time segment
        :return: generator generating successive realizations of the process
        """
        _point = np.zeros(shape=(1,)) if start_point is None else start_point
        size = _point.shape
        for _ in range(int(time * division)):
            yield _point
            _point = _point + np.random.poisson(lam / division, size=size)

    @staticmethod
    def generate_one_dim_jumps(lam=1, time=1):
        """Generates the one dimensional process on the interval [0, time]
        :param lam: float, the intensity of the process
        :param time: time interval
        :return: numpy.array with jumps of the process
        """
        jumps = [0.0]
        while jumps[-1] < time:
            jumps.append(jumps[-1] + np.random.exponential(scale=1 / lam))
        return np.array(jumps)

    @staticmethod
    def generate_jumps(lam=1, dim=1, time=1):
        """Generates the process on the interval [0, time]
        :param lam: float, the intensity of the process
        :param dim: dimensionality of the process
        :param time: time interval
        :return: generator generating successive realizations of the process
        """
        jumps = []
        for _ in range(dim):
            jumps.append(Poisson.generate_one_dim_jumps(lam=lam, time=time))
        return jumps

    @staticmethod
    def generate_on_grid(_grid, lam=1, start_point=None):
        """Function generating the process on grid
        :param _grid: sorted list or array with grid
        :param lam: float, the intensity of the process
        :param start_point: numpy.array with shape (n,),
         starting point of the process
        :return: array with realizations of the Poisson process on grid
         """
        _arr = [np.zeros(shape=(1,))] if start_point is None else start_point
        for i in range(1, len(_grid), 1):
            _arr.append(_arr[-1] + np.random.poisson(
                lam=lam * (_grid[i] - _grid[i - 1]), size=start_point.shape))
        return np.array(_arr).transpose()
