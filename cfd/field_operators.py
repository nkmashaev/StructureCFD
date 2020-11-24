from typing import Any, Callable, Tuple

import numpy as np
from numpy import linalg as LA

import cfd.functions as func


def with_iteration(numb: int) -> Callable[[Any], Any]:
    def decorator(func: Callable[[Any], Any]):
        # @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(numb):
                func(*args, **kwargs)

        return wrapper

    return decorator


@with_iteration(numb=15)
def green_gauss(
    i_size: int,
    j_size: int,
    var: np.ndarray,
    grad: np.ndarray,
    cell_center_arr: np.ndarray,
    cell_volume_arr: np.ndarray,
    i_face_center_arr: np.ndarray,
    i_face_vector_arr: np.ndarray,
    j_face_center_arr: np.ndarray,
    j_face_vector_arr: np.ndarray,
):
    n_cell = np.zeros((4, 2), dtype=int)
    r_f = np.zeros((4, 2))
    s_f = np.zeros((4, 2))
    r_c = np.zeros((4, 2))
    vol = 0.0
    grad_c = np.zeros(2)
    grad_m = np.zeros(2)
    for i in range(1, i_size):
        for j in range(1, j_size):
            n_cell[0] = np.array([i - 1, j], dtype=int)
            n_cell[1] = np.array([i + 1, j], dtype=int)
            n_cell[2] = np.array([i, j - 1], dtype=int)
            n_cell[3] = np.array([i, j + 1], dtype=int)

            r_f[0] = i_face_center_arr[i - 1, j - 1]
            r_f[1] = i_face_center_arr[i, j - 1]
            r_f[2] = j_face_center_arr[i - 1, j - 1]
            r_f[3] = j_face_center_arr[i - 1, j]

            s_f[0] = -i_face_vector_arr[i - 1, j - 1]
            s_f[1] = i_face_vector_arr[i, j - 1]
            s_f[2] = -j_face_vector_arr[i - 1, j - 1]
            s_f[3] = j_face_vector_arr[i - 1, j]

            vol = cell_volume_arr[i - 1, j - 1]
            r_c = cell_center_arr[i, j]

            grad_c = np.zeros(2)
            for i_face, neighbour in enumerate(n_cell):
                i_n = neighbour[0]
                j_n = neighbour[1]
                r_n = cell_center_arr[i_n][j_n]
                d_c = LA.norm(r_f[i_face] - r_c)
                d_n = LA.norm(r_f[i_face] - r_n)

                var_m = func.linear_interpolation(d_c, d_n, var[i][j], var[i_n][j_n])
                r_m = np.array(
                    [
                        func.linear_interpolation(
                            d_c,
                            d_n,
                            cell_center_arr[i][j][k],
                            cell_center_arr[i_n][j_n][k],
                        )
                        for k in range(2)
                    ]
                )
                grad_m = np.array(
                    [
                        func.linear_interpolation(
                            d_c, d_n, grad[i][j][k], grad[i_n][j_n][k]
                        )
                        for k in range(2)
                    ]
                )
                var_f = var_m + np.dot(r_f[i_face] - r_m, grad_m)
                grad_c = grad_c + var_f * s_f[i_face]
            grad[i][j] = grad_c / vol


def divergence(
    i_size: int,
    j_size: int,
    var: np.ndarray,
    div: np.ndarray,
    scalar: np.ndarray,
    grad_scalar: np.ndarray,
    cell_center_arr: np.ndarray,
    cell_volume_arr: np.ndarray,
    i_face_center_arr: np.ndarray,
    i_face_vector_arr: np.ndarray,
    j_face_center_arr: np.ndarray,
    j_face_vector_arr: np.ndarray,
    mode: int = 0,
):
    n_cell = np.zeros((4, 2), dtype=int)
    r_f = np.zeros((4, 2))
    s_f = np.zeros((4, 2))
    r_c = np.zeros((4, 2))
    vol = 0.0
    counter = 0
    for i in range(1, i_size):
        for j in range(1, j_size):
            n_cell[0] = np.array([i - 1, j], dtype=int)
            n_cell[1] = np.array([i + 1, j], dtype=int)
            n_cell[2] = np.array([i, j - 1], dtype=int)
            n_cell[3] = np.array([i, j + 1], dtype=int)

            r_f[0] = i_face_center_arr[i - 1, j - 1]
            r_f[1] = i_face_center_arr[i, j - 1]
            r_f[2] = j_face_center_arr[i - 1, j - 1]
            r_f[3] = j_face_center_arr[i - 1, j]

            s_f[0] = -i_face_vector_arr[i - 1, j - 1]
            s_f[1] = i_face_vector_arr[i, j - 1]
            s_f[2] = -j_face_vector_arr[i - 1, j - 1]
            s_f[3] = j_face_vector_arr[i - 1, j]

            vol = cell_volume_arr[i - 1, j - 1]
            r_c = cell_center_arr[i, j]

            for i_face, neighbour in enumerate(n_cell):
                i_n = neighbour[0]
                j_n = neighbour[1]
                r_n = cell_center_arr[i_n][j_n]
                d_c = LA.norm(r_f[i_face] - r_c)
                d_n = LA.norm(r_f[i_face] - r_n)

                var_f = np.array(
                    [
                        func.linear_interpolation(
                            d_c, d_n, var[i][j][k], var[i_n][j_n][k]
                        )
                        for k in range(2)
                    ]
                )
                if mode == 0:
                    div[i][j] = div[i][j] + np.dot(var_f, s_f[i_face])
                elif mode == 1:
                    scalar_f = func.linear_interpolation(
                        d_c, d_n, scalar[i][j], scalar[i_n][j_n]
                    )
                    div[i][j] = div[i][j] + np.dot(scalar_f * var_f, s_f[i_face])
                elif mode == 2:
                    scalar_f = 0.0
                    is_boundary = d_n < 1.0e-6

                    if np.dot(var_f, s_f[i_face]) >= 0:
                        scalar_f = scalar[i][j]
                    else:
                        if is_boundary:
                            counter += 1
                            scalar_f = 2.0 * scalar[i_n][j_n] - scalar[i][j]
                        else:
                            scalar_f = scalar[i_n][j_n]

                    div[i][j] = div[i][j] + np.dot(scalar_f * var_f, s_f[i_face])
                elif mode == 3:
                    scalar_f = 0.0

                    is_boundary = d_n < 1.0e-6

                    if np.dot(var_f, s_f[i_face]) >= 0:
                        scalar_f = scalar[i][j] + np.dot(
                            grad_scalar[i][j], r_f[i_face] - r_c
                        )
                    else:
                        if is_boundary:
                            scalar_f_fo = 2.0 * scalar[i_n][j_n] - scalar[i][j]
                            grad_c_dot_r = np.dot(grad_scalar[i][j], r_c - r_f[i_face])
                            grad_b_dot_r = scalar[i][j] - scalar[i_n][j_n]
                            grad_d_dot_r = 4.0 * grad_b_dot_r - 3.0 * grad_c_dot_r
                            scalar_f = scalar_f_fo + grad_d_dot_r
                        else:
                            scalar_f = scalar[i_n][j_n] + np.dot(
                                grad_scalar[i_n][j_n], r_f[i_face] - r_n
                            )

                    div[i][j] = div[i][j] + np.dot(scalar_f * var_f, s_f[i_face])
            div[i][j] = div[i][j] / vol
    print(counter)


def curl(
    i_size: int,
    j_size: int,
    var: np.ndarray,
    curl_z: np.ndarray,
    cell_center_arr: np.ndarray,
    cell_volume_arr: np.ndarray,
    i_face_center_arr: np.ndarray,
    i_face_vector_arr: np.ndarray,
    j_face_center_arr: np.ndarray,
    j_face_vector_arr: np.ndarray,
):
    n_cell = np.zeros((4, 2), dtype=int)
    r_f = np.zeros((4, 2))
    s_f = np.zeros((4, 2))
    r_c = np.zeros((4, 2))
    vol = 0.0
    for i in range(1, i_size):
        for j in range(1, j_size):
            n_cell[0] = np.array([i - 1, j], dtype=int)
            n_cell[1] = np.array([i + 1, j], dtype=int)
            n_cell[2] = np.array([i, j - 1], dtype=int)
            n_cell[3] = np.array([i, j + 1], dtype=int)

            r_f[0] = i_face_center_arr[i - 1, j - 1]
            r_f[1] = i_face_center_arr[i, j - 1]
            r_f[2] = j_face_center_arr[i - 1, j - 1]
            r_f[3] = j_face_center_arr[i - 1, j]

            s_f[0] = -i_face_vector_arr[i - 1, j - 1]
            s_f[1] = i_face_vector_arr[i, j - 1]
            s_f[2] = -j_face_vector_arr[i - 1, j - 1]
            s_f[3] = j_face_vector_arr[i - 1, j]

            vol = cell_volume_arr[i - 1, j - 1]
            r_c = cell_center_arr[i, j]

            for i_face, neighbour in enumerate(n_cell):
                i_n = neighbour[0]
                j_n = neighbour[1]
                r_n = cell_center_arr[i_n][j_n]
                d_c = LA.norm(r_f[i_face] - r_c)
                d_n = LA.norm(r_f[i_face] - r_n)

                var_f = np.array(
                    [
                        func.linear_interpolation(
                            d_c, d_n, var[i][j][k], var[i_n][j_n][k]
                        )
                        for k in range(2)
                    ]
                )
                curl_z[i][j] = curl_z[i][j] + np.cross(var_f, s_f[i_face])
            curl_z[i][j] = curl_z[i][j] / vol


def laplacian(
    i_size: int,
    j_size: int,
    var: np.ndarray,
    laplacian_op: np.ndarray,
    grad: np.ndarray,
    cell_center_arr: np.ndarray,
    cell_volume_arr: np.ndarray,
    i_face_center_arr: np.ndarray,
    i_face_vector_arr: np.ndarray,
    j_face_center_arr: np.ndarray,
    j_face_vector_arr: np.ndarray,
):
    n_cell = np.zeros((4, 2), dtype=int)
    r_f = np.zeros((4, 2))
    s_f = np.zeros((4, 2))
    counter = 0
    for i in range(1, i_size):
        for j in range(1, j_size):
            n_cell[0] = np.array([i - 1, j], dtype=int)
            n_cell[1] = np.array([i + 1, j], dtype=int)
            n_cell[2] = np.array([i, j - 1], dtype=int)
            n_cell[3] = np.array([i, j + 1], dtype=int)

            r_f[0] = i_face_center_arr[i - 1, j - 1, :]
            r_f[1] = i_face_center_arr[i, j - 1, :]
            r_f[2] = j_face_center_arr[i - 1, j - 1, :]
            r_f[3] = j_face_center_arr[i - 1, j, :]

            s_f[0] = -i_face_vector_arr[i - 1, j - 1, :]
            s_f[1] = i_face_vector_arr[i, j - 1, :]
            s_f[2] = -j_face_vector_arr[i - 1, j - 1, :]
            s_f[3] = j_face_vector_arr[i - 1, j, :]

            vol = cell_volume_arr[i - 1, j - 1]
            r_c = cell_center_arr[i, j, :]

            for i_face, neighbour in enumerate(n_cell):
                i_n = neighbour[0]
                j_n = neighbour[1]
                r_n = cell_center_arr[i_n, j_n, :]
                d_c = LA.norm(r_f[i_face, :] - r_c[:])
                d_n = LA.norm(r_f[i_face, :] - r_n[:])
                d_n_c = LA.norm(r_n[:] - r_c[:])
                n_f = s_f[i_face, :] / LA.norm(s_f[i_face, :])
                dp_dn = (var[i_n, j_n] - var[i, j]) / d_n_c

                r_n_c = (r_n[:] - r_c[:]) / d_n_c
                grad_f = np.array(
                    [
                        func.linear_interpolation(
                            d_c, d_n, grad[i, j, k], grad[i_n, j_n, k]
                        )
                        for k in range(2)
                    ]
                )
                if d_n < 1.0e-6:
                    counter += 1
                    dp_dn_c = np.dot(grad[i, j, :], n_f[:])
                    print(f"grad[i,j] = {grad[i,j,:]}")
                    print(f"n_f = {n_f[:]}")
                    print(f"dp_dn_c = {dp_dn_c}")

                    dp_dn = (5.0 * dp_dn - 2.0 * dp_dn_c) / 3.0
                    grad_f[:] = grad[i, j, :]

                dp_dn = dp_dn + np.dot(n_f[:] - r_n_c[:], grad_f[:])
                laplacian_op[i, j] += dp_dn * LA.norm(s_f[i_face, :])
            laplacian_op[i, j] /= vol
    print(counter)
