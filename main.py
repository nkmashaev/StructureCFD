import os
from typing import Tuple

import numpy as np

import cfd.field_operators as fop
import cfd.functions as func
import gridtools.grid_manager as gm

dim = 2
mode = 3
file_name = os.path.join("grids", "base.msh")

x_arr, y_arr = gm.read_grid(file_name)
i_size, j_size = x_arr.shape

cell_center_arr = np.zeros((i_size + 1, j_size + 1, dim))
velocity_arr = np.zeros((i_size + 1, j_size + 1, dim))
div_velocity = np.zeros((i_size + 1, j_size + 1))
div_exact = np.zeros((i_size + 1, j_size + 1))
div_error = np.zeros((i_size + 1, j_size + 1))
grad_pressure = np.zeros((i_size + 1, j_size + 1, dim))
grad_exact = np.zeros((i_size + 1, j_size + 1, dim))
grad_error = np.zeros((i_size + 1, j_size + 1, dim))
pressure_arr = np.zeros((i_size + 1, j_size + 1))
cell_volume_arr = np.zeros((i_size - 1, j_size - 1))
i_face_center_arr = np.zeros((i_size, j_size - 1, dim))
i_face_vector_arr = np.zeros((i_size, j_size - 1, dim))
j_face_center_arr = np.zeros((i_size - 1, j_size, dim))
j_face_vector_arr = np.zeros((i_size - 1, j_size, dim))
gm.calculate_metric(
    i_size,
    j_size,
    x_arr,
    y_arr,
    cell_center_arr,
    cell_volume_arr,
    i_face_center_arr,
    i_face_vector_arr,
    j_face_center_arr,
    j_face_vector_arr,
)

print("Initiate fields")
for i in range(i_size + 1):
    for j in range(j_size + 1):
        pressure_arr[i][j] = func.scalar_center_init(
            cell_center_arr[i][j][0], cell_center_arr[i][j][1]
        )
        grad_exact[i][j] = func.grad_center_init(
            cell_center_arr[i][j][0], cell_center_arr[i][j][1]
        )
        velocity_arr[i][j] = func.vector_center_init(
            cell_center_arr[i][j][0], cell_center_arr[i][j][1]
        )
        div_exact[i][j] = func.div_center_init(
            cell_center_arr[i][j][0], cell_center_arr[i][j][1], mode
        )

iter_numb = 15
for i in range(iter_numb):
    fop.green_gauss(
        i_size,
        j_size,
        pressure_arr,
        grad_pressure,
        cell_center_arr,
        cell_volume_arr,
        i_face_center_arr,
        i_face_vector_arr,
        j_face_center_arr,
        j_face_vector_arr,
    )
    grad_error = abs(grad_exact - grad_pressure) / grad_exact
    max_grad_err = np.max(grad_error[1:i_size, 1:j_size])
    print(f"Maximum grad_error is {max_grad_err:.11E}")


fop.divergence(
    i_size,
    j_size,
    velocity_arr,
    div_velocity,
    pressure_arr,
    grad_pressure,
    cell_center_arr,
    cell_volume_arr,
    i_face_center_arr,
    i_face_vector_arr,
    j_face_center_arr,
    j_face_vector_arr,
    mode,
)
div_error = abs(div_exact - div_velocity) / div_exact
max_div_err = np.amax(div_error[1:i_size, 1:j_size])
print(f"Maximum div_error is {max_div_err:.11E}")

gm.output_data(
    "data.plt",
    i_size,
    j_size,
    x_arr,
    y_arr,
    pressure_arr,
    velocity_arr,
    grad_pressure,
    grad_error,
    div_velocity,
    div_error,
)
