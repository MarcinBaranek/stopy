from stopy.dif_equ import ItoProcess
from stopy.montecarlo import IntegrateMonteCarlo, MonteCarlo
import logging


def logger__init__(name="logger", log_path="logs.log"):
    # create logger
    logging.basicConfig(
        filename=log_path, datefmt="%H:%M:%S", level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", "%H:%M:%S")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger


""" Aim is implemented Feynman-Kac formula
    (du / dt)(x, t)
    + mu(x, t) * (du / dx)(x, t)
    + 0.5 * sigma^2(x, t) * (d^2u / d^2x)(x, t)
    - V(x, t) * u(x, t)
    + f(x, t)
    = 0.0
    with conditional
    u(x, T) = psi(x)
    ~~~~~~~~~~~~~~~~~~~~
    I(V) = integrate from t to T with function V(x_tau, tau) d tau
    I(g) = integrate from t to T with function g(x_r, r) dr
    u(x, t) = E{
                I( Exp( - I(V)) f(X_r, r))
                + Exp ( - I(V)) ) * psi(X_T)
                | X_t = x
                }
"""


class MarginalIntegral:
    def __init__(self, func):
        self.value = None,
        self.func = func,
        self.weight = 0


class ExpectedValue:
    def __init__(self, ):
        pass


class FeynmanKac:
    def __init__(self, a_func=None, b_func=None, v_func=None, bias=None,
                 bound_func=None, end_time=1.0, log_path: str = ""):
        """
        # ToDO write the documentation
        :param a_func:
        :param b_func:
        :param v_func:
        :param bias:
        :param bound_func:
        :param end_time:
        :param log_path:
        :param bound_func:
        """
        self.process = ItoProcess(a_func, b_func)
        self.v_func = v_func if v_func is not None else lambda x, t: 0.0
        self.bias = bias if bias is not None else lambda x, t: 0.0
        self.bound_func = lambda x: 0.0 if bound_func is None else bound_func
        self.simulation_params = None
        self.end_time = end_time
        if log_path:
            logger = logger__init__(name="FeynmanKac", log_path=log_path)
            logger.info("object FeynmanKac has been successfully created")

    """ Param dict
    {
        "process": 
        {
            "schema": Eulerschema,
            "generate params": 
            {
                "point": 
                "t_0": 0.0
                "dt": 1.e-8
            }
            "condition of end": # generated T / dt elements of Ito process
        },
        "V Integral":
        {
            "t": 0.0
            "N trajectory": 100,
            "M steps": 100,
            "f Integral": True,
            "Psi Integral": True
            # ToDo add more advances options
        }
        "f Integral":
        {
            "t": 0.0
            "N trajectory": 100,
            "M steps": 100,
        }
    }
    """

    def set_params_dict(self, *args, **kwargs):
        # ToDo write this continuous
        self.simulation_params["N_simulations"] = args
        self.simulation_params["V Integral"] = kwargs

    def __compute_v_integral(self, init):
        # ToDo write this ASAP
        pass
