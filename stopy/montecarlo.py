import numpy as np
import scipy.stats as st
import traceback


class MonteCarlo:
    """Monte Carlo basic class for carrying out monte carlo simulations.
    A class is not useful for practical tasks,
    it exists because it is the foundation for subsequent classes
    """

    def __init__(self, func=None):
        """Constructor, only assigns the functions to be 'averaged'
        :param func: functions, Identity is assigned by default
        """
        self.func = func if func is not None else lambda x: x

    @staticmethod
    def mean(func, param: dict = None, n_sim=10):
        """
        # ToDo write the documentation
        :param func:
        :param param:
        :param n_sim:
        :return:
        """
        return sum(func(**param) for _ in range(n_sim)) / n_sim if param else\
            sum(func() for _ in range(n_sim)) / n_sim

    @staticmethod
    def mean_with_variance(func, params: dict = None, n_sim=10):
        """
        # ToDo write the documentation
        :param func:
        :param params:
        :param n_sim:
        :return: avg, var
        """
        sample = [func(**params) for _ in range(n_sim)] if params else\
            [func() for _ in range(n_sim)]
        avg = sum(sample) / n_sim
        var = sum((_x - avg) ** 2 for _x in sample) / (n_sim - 1)
        return avg, var


class IntegrateMonteCarlo(MonteCarlo):
    def __init__(self, func=None, gen=None):
        """
        # ToDO write the documentation
        :param func: functions, Identity is assigned by default
        :param gen:
        """
        super().__init__(func)
        self.gen = gen if gen is not None else np.random.rand
        self.avg = None
        self.var = None
        self.n_samples = None   # n_sim of data

    def simulate(self, n_sim=10, var=False):
        """
        # ToDO write the documentation
        :param n_sim: int > 0, number of samples
        :param var: bool, if true then compute also variance else only mean
        :return: None
        """
        if var:
            self.avg, self.var = self.mean_with_variance(self.func(self.gen))
            self.n_samples = n_sim
        else:
            self.avg = self.mean(self.func(self.gen))
            self.n_samples = n_sim

    def confidence_interval(self, alpha=0.05):
        """
        # ToDO write the documentation
        :param alpha:
        :return:
        """
        if self.var is None:
            self.simulate()
        margin = st.norm.interval(1.0 - alpha)[1]\
            * np.sqrt(self.var / self.n_samples)
        return self.avg - margin, self.avg + margin
