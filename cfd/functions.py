import numpy as np


def linear_interpolation(
    dist1: float,
    dist2: float,
    val1: float,
    val2: float,
) -> float:
    interpolated_val = (val1 * dist2 + val2 * dist1) / (dist1 + dist2)
    return interpolated_val


def scalar_center_init(x: float, y: float) -> float:
    init_val = x + y
    return init_val


def grad_center_init(x: float, y: float) -> np.ndarray:
    exact_grad = np.array((1.0, 1.0))
    return exact_grad


def vector_center_init(x: float, y: float) -> np.ndarray:
    init_vect = np.array(((1.0 + x) * (1.0 + y), x * y))
    return init_vect


def div_center_init(x: float, y: float, mode: int = 0) -> float:
    if mode == 0:
        exact_div = 1.0 + x + y
    else:
        exact_div = x ** 2 + 4 * x * y + 2 * x + y ** 2 + 2 * y + 1.0
    return exact_div


def curl_center_init(x: float, y: float) -> float:
    exact_curl = 1.0 + x - y
    return exact_curl
