from collections import namedtuple
from typing import Any, Callable, List

import numpy as np

init = namedtuple("init_func", ["p", "T", "u", "v"])


def linear_interpolation(d1: float, d2: float, x1: float, x2: float) -> float:
    """
    linear_interpolation function is designed to reconstruct value using another
    to variables x1, x2 with their weights d1,d2
    :param d1: first value x1 weight
    :param d2: second value x2 weight
    :param x1: first variable
    :param x2: second variable
    :return: restored variable
    """
    interpolated_val = (x1 * d2 + x2 * d1) / (d1 + d2)
    return interpolated_val


def coord_deg_sum(n: int) -> Callable[[Any], Any]:
    def coord_sum(*args: float) -> float:
        val = 0.0
        for x_i in args:
            val += x_i ** n
        return val

    return coord_sum


def coord_sum_deg(n: int) -> Callable[[Any], Any]:
    def coord_sum(*args: float) -> float:
        val = 0.0
        for x_i in args:
            val += x_i
        return val ** n

    return coord_sum


def component_deg(i: int, n: int) -> Callable[[Any], Any]:
    def component(*args: float) -> float:
        return args[i] ** n

    return component


def constant_val(const_val: float) -> Callable[[Any], Any]:
    def coord_call(*args: float) -> float:
        return const_val

    return coord_call


def coord_mult_deg(n: int, xn: List[float] = None):
    def mult_init(*args: float) -> float:
        nonlocal xn
        if xn is None:
            xn = [0.0] * len(args)
        if len(args) != len(xn):
            raise (AttributeError("Error: Parameter dimension mismatch!"))
        val = 1.0
        for i, x_i in enumerate(args):
            val *= (xn[i] + x_i) ** n
        return val

    return mult_init


# def scalar_center_init(x: float, y: float) -> float:
#     #init_val = x + y
#     init_val = x * x + y * y
#     return init_val
#
#
# def grad_center_init(x: float, y: float) -> np.ndarray:
#     exact_grad = np.array((1.0, 1.0))
#     #exact_grad = np.array((2.0 * x, 2.0 * y))
#     return exact_grad
#
#
# def vector_center_init(x: float, y: float) -> np.ndarray:
#     init_vect = np.array(((1.0 + x) * (1.0 + y), x * y))
#     return init_vect
#
#
# def div_center_init(x: float, y: float, mode: int = 0) -> float:
#     if mode == 0:
#         exact_div = 1.0 + x + y
#     else:
#         exact_div = x ** 2 + 4 * x * y + 2 * x + y ** 2 + 2 * y + 1.0
#     return exact_div
#
#
# def curl_center_init(x: float, y: float) -> float:
#     exact_curl = 1.0 + x - y
#     return exact_curl
#
# def laplacian_center_init(x: float, y: float) -> float:
#     exact_laplacian = 4.0
#     return exact_laplacian
