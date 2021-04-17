import numpy as np
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as st
import traceback


class MonteCarlo:
    def __init__(self, func=None):
        self.func = func if func is not None else lambda x: x

    @staticmethod
    def iter_arg(n_yield=1, *args, **kwargs):
        for _ in range(n_yield):
            yield args, kwargs

    @staticmethod
    def iter_gen(n_yield=1, gen=np.random.rand, *args, **kwargs):
        for _ in range(n_yield):
            yield gen(args, kwargs)

    @staticmethod
    def fast_mean(func, param, n_sim=10, workers=3):
        param = param if isinstance(param, Iterable) else\
            MonteCarlo.iter_arg(n_sim, param)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(func, param)
        return np.array([result for result in results]).mean()

    @staticmethod
    def fast_mean_with_variance(func, sample, workers=3):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            avg = np.array(executor.map(func, sample))
            var = np.array(executor.map(
                lambda x: (func(x) - avg)**2), sample).sum() / (len(sample) - 1)
            return avg, var


class IntegrateMonteCarlo(MonteCarlo):
    def __init__(self, func=None, gen=None):
        super().__init__(func)
        self.gen = gen if gen is not None else np.random.rand
        self.mean = None
        self.var = None
        self.n_samples = None

    def simulate(self, size=10, var=False, workers=3):
        if var:
            sample = np.array([self.gen() for _ in range(size)])
            self.mean, self.var =\
                self.fast_mean_with_variance(self.func, sample, workers=workers)
            self.n_samples = size
        else:
            self.mean = self.fast_mean(self.func, self.iter_gen(size, self.gen),
                                       workers=workers)

    def confidence_interval(self, alpha=0.05):
        try:
            margin = st.norm.interval(1.0 - alpha) *\
                      np.sqrt(self.var / self.n_samples)
            return self.mean - margin, self.mean + margin
        except AttributeError as e:
            print(e)
            traceback.print_exc()
            print("make sure you run simulations and calculate variance")
