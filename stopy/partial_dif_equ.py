from stopy.dif_equ import ItoProcess
from stopy.montecarlo import IntegrateMonteCarlo, MonteCarlo

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
                I( Exp( - I(V) f(X_r, r))
                + Exp ( - I(V)) ) * psi(X_T)
                | X_t = x
                }
"""


class FeynmanKac(ItoProcess):
    def __init__(self, a_func=None, b_func=None, v_func=None, bias=None,
                 bound_func=None):
        """
        # ToDO write the documentation
        :param a_func:
        :param b_func:
        :param v_func:
        :param bias:
        :param bound_func:
        """
        super().__init__(a_func, b_func)
        self.v_func = v_func if v_func is not None else lambda x, t: 0.0
        self.bias = bias if bias is not None else lambda x, t: 0.0
        self.bound_func = lambda x: 0.0 if bound_func is None else bound_func
        self.__params_dict = None
        self.results = []

    def set_params_dict(self, *args, **kwargs):
        self.__params_dict["N_simulations"] = args
        self.__params_dict["V Integral"] = kwargs

    def __compute_v_integral(self, init):
        pass
