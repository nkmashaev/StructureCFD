from typing import Tuple

import numpy as np
from numpy import linalg as LA


def read_grid(mesh_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read structured grid nodes data from file with mesh_name

    :param mesh_name: name of grid file
    :return: arrays of x and y coordinates of nodes
    """
    with open(mesh_name, "r") as in_file:
        i_size, j_size = map(int, in_file.readline().split())
        x_arr = np.zeros((i_size, j_size))
        y_arr = np.zeros((i_size, j_size))

        for i in range(i_size):
            for j in range(j_size):
                x_arr[i][j], y_arr[i][j] = map(float, in_file.readline().split())

    return x_arr, y_arr


def calculate_metric(
    i_size: int,
    j_size: int,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    cell_center_arr: np.ndarray,
    cell_volume_arr: np.ndarray,
    i_face_center_arr: np.ndarray,
    i_face_vector_arr: np.ndarray,
    j_face_center_arr: np.ndarray,
    j_face_vector_arr: np.ndarray,
):
    # i-direction faces initialization
    for i in range(i_size):
        for j in range(j_size - 1):
            x = x_arr[i][j + 1] - x_arr[i][j]
            y = y_arr[i][j + 1] - y_arr[i][j]
            i_face_vector_arr[i][j][0] = y
            i_face_vector_arr[i][j][1] = -x
            i_face_center_arr[i][j][0] = 0.5 * (x_arr[i][j + 1] + x_arr[i][j])
            i_face_center_arr[i][j][1] = 0.5 * (y_arr[i][j + 1] + y_arr[i][j])

    # j-direction faces initialization
    for i in range(i_size - 1):
        for j in range(j_size):
            x = x_arr[i + 1][j] - x_arr[i][j]
            y = y_arr[i + 1][j] - y_arr[i][j]
            j_face_vector_arr[i][j][0] = -y
            j_face_vector_arr[i][j][1] = x
            j_face_center_arr[i][j][0] = 0.5 * (x_arr[i + 1][j] + x_arr[i][j])
            j_face_center_arr[i][j][1] = 0.5 * (y_arr[i + 1][j] + y_arr[i][j])

    # cell volumes
    radius_vector = np.zeros(2)
    for i in range(0, i_size - 1):
        for j in range(0, j_size - 1):
            radius_vector[0] = x_arr[i + 1][j + 1] - x_arr[i][j]
            radius_vector[1] = y_arr[i + 1][j + 1] - y_arr[i][j]
            cell_volume_arr[i][j] = 0.5 * (
                np.dot(i_face_vector_arr[i][j], radius_vector)
                + np.dot(j_face_vector_arr[i][j], radius_vector)
            )

    # cell centers
    for i in range(1, i_size):
        for j in range(1, j_size):
            cell_center_arr[i][j] = (
                i_face_center_arr[i - 1][j - 1]
                * LA.norm(i_face_vector_arr[i - 1][j - 1])
                + i_face_center_arr[i][j - 1] * LA.norm(i_face_vector_arr[i][j - 1])
                + j_face_center_arr[i - 1][j - 1]
                * LA.norm(j_face_vector_arr[i - 1][j - 1])
                + j_face_center_arr[i - 1][j] * LA.norm(j_face_vector_arr[i - 1][j])
            )
            cell_center_arr[i][j] /= (
                LA.norm(i_face_vector_arr[i - 1][j - 1])
                + LA.norm(i_face_vector_arr[i][j - 1])
                + LA.norm(j_face_vector_arr[i - 1][j - 1])
                + LA.norm(j_face_vector_arr[i - 1][j])
            )

    for j in range(1, j_size):
        cell_center_arr[0][j] = i_face_center_arr[0][j - 1]
        cell_center_arr[i_size][j] = i_face_center_arr[i_size - 1][j - 1]

    for i in range(1, i_size):
        cell_center_arr[i][0] = j_face_center_arr[i - 1][0]
        cell_center_arr[i][j_size] = j_face_center_arr[i - 1][j_size - 1]


def output_data(
    file_path: str,
    i_size: int,
    j_size: int,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray,
    grad_pressure: np.ndarray,
    grad_error: np.ndarray,
    div_velocity: np.ndarray,
    div_error: np.ndarray,
):
    with open(file_path, "w") as out_file:
        out_file.write(
            "VARIABLES ="
            ' "X", '
            ' "Y", '
            ' "Pressure", '
            ' "U", '
            ' "V", '
            ' "PressureGradientX", '
            ' "PressureGradientY", '
            ' "PressureGradientErrX", '
            ' "PressureGradientErrY", '
            ' "VelocityDivergence", '
            ' "VelocityDivergenceErr"\n'
        )
        out_file.write(
            f"ZONE I={i_size}, J={j_size},"
            "DATAPACKING=BLOCK, VARLOCATION=([3-11]=CELLCENTERED)\n"
        )
        for i in range(i_size):
            for j in range(j_size):
                out_file.write(f"{x_arr[i][j]:.11E} ")
        out_file.write("\n")

        for i in range(i_size):
            for j in range(j_size):
                out_file.write(f"{y_arr[i][j]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{pressure[i][j]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{velocity[i][j][0]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{velocity[i][j][1]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{grad_pressure[i][j][0]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{grad_pressure[i][j][1]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{grad_error[i][j][0]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{grad_error[i][j][1]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{div_velocity[i][j]:.11E} ")
        out_file.write("\n")

        for i in range(1, i_size):
            for j in range(1, j_size):
                out_file.write(f"{div_error[i][j]:.11E} ")
        out_file.write("\n")
