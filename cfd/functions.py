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
    init_vect = np.array((1.0 + x, 1.0 + y))
    return init_vect


def div_center_init(x: float, y: float, mode: int = 0) -> float:
    if mode == 0:
        exact_div = 2.0
    else:
        exact_div = 2.0 + 3.0 * x + 3.0 * y
    return exact_div
