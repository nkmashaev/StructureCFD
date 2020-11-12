import os
from typing import Tuple

import numpy as np

import cfd.field_operators as fop
import cfd.functions as func
import gridtools.grid_manager as gm

# pre_init
scheme_dict = {
    0: "vector divergence [ div(V)]",
    1: "central scheme [ div(pV) ]",
    2: "first order upwind [ div(pV) ]",
    3: "second order upwind [ div(pV) ]",
}
dim = 2

# reading initialize.txt file
with open("initialize.txt", "r") as in_file:
    file_name = in_file.readline().split("#")[0].strip()
    iter_numb = int(in_file.readline().split("#")[0])
    mode = int(in_file.readline().split("#")[0])
    assert 0 <= mode <= 3

# print input parameters
print("=" * 30)
print("Info:")
print(f"Mesh file name: {file_name}")
print(f"Number of Green Gauss iterations: {iter_numb}")
print(f"Divergence mode calculation approach: {mode} - {scheme_dict[mode]}")
print("=" * 30)

# reading mesh file
print(f"Reading {file_name} mesh file...")
x_arr, y_arr = gm.read_grid(file_name)
i_size, j_size = x_arr.shape

# initializing of arrays
# cell_center_arr - array of cells centers. Allocating extra variables for dummy boundary cells
# There is a shift of indices relative to cell_volume_array due to lack of volumes magnitude for dummies
cell_center_arr = np.zeros((i_size + 1, j_size + 1, dim))

# cell_volume_arr - array of cells' volumes. Only inner volumes are storaged - it is the cause of
# inconsistency in a cell_center_arr and a cell_volume_arr indices
cell_volume_arr = np.zeros((i_size - 1, j_size - 1))

# velocity_arr - array of velocity vectors' components.
velocity_arr = np.zeros((i_size + 1, j_size + 1, dim))

div_velocity = np.zeros((i_size + 1, j_size + 1))
div_exact = np.zeros((i_size + 1, j_size + 1))
div_error = np.zeros((i_size + 1, j_size + 1))
curl_z_velocity = np.zeros((i_size + 1, j_size + 1))
curl_z_exact = np.zeros((i_size + 1, j_size + 1))
curl_z_error = np.zeros((i_size + 1, j_size + 1))
grad_pressure = np.zeros((i_size + 1, j_size + 1, dim))
grad_exact = np.zeros((i_size + 1, j_size + 1, dim))
grad_error = np.zeros((i_size + 1, j_size + 1, dim))
pressure_arr = np.zeros((i_size + 1, j_size + 1))
i_face_center_arr = np.zeros((i_size, j_size - 1, dim))
i_face_vector_arr = np.zeros((i_size, j_size - 1, dim))
j_face_center_arr = np.zeros((i_size - 1, j_size, dim))
j_face_vector_arr = np.zeros((i_size - 1, j_size, dim))

# metric calculation
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

# initialization of fields
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
        curl_z_exact[i][j] = func.curl_center_init(
            cell_center_arr[i][j][0], cell_center_arr[i][j][1]
        )

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

fop.curl(
    i_size,
    j_size,
    velocity_arr,
    curl_z_velocity,
    cell_center_arr,
    cell_volume_arr,
    i_face_center_arr,
    i_face_vector_arr,
    j_face_center_arr,
    j_face_vector_arr,
)
curl_z_error = abs(curl_z_exact - curl_z_velocity) / curl_z_exact
max_rot_z_err = np.amax(curl_z_error[1:i_size, 1:j_size])
print(f"Maximum rotor_z_error is {max_rot_z_err:.11E}")

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
    curl_z_velocity,
    curl_z_error,
)
