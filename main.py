import functools
import os
import re
from collections import namedtuple
from typing import Any, Callable, Tuple

import numpy as np
from numpy import linalg as LA

import cfd.functions as func
import gridtools.grid_manager as gm

init_dict = {}


# pre_init
scheme_dict = {
    1: "central scheme [ div(pV) ]",
    2: "first order upwind [ div(pV) ]",
    3: "second order upwind [ div(pV) ]",
}
dim = 2

# reading initialize.txt file
with open("initialize.txt", "r") as in_file:
    file_name = in_file.readline().split("#")[0].strip()
    grad_mode = int(in_file.readline().split("#")[0].strip())
    assert 0 <= grad_mode <= 1
    iter_numb = int(in_file.readline().split("#")[0])
    mode = int(in_file.readline().split("#")[0])
    assert 1 <= mode <= 3

# print input parameters
print("=" * 60)
print("Info:")
print(f"Mesh file name: {file_name}")
print(f"Number of Green Gauss iterations: {iter_numb}")
print(f"Divergence mode calculation approach: {mode} - {scheme_dict[mode]}")
print("=" * 60)

# reading mesh file
print(f"Reading {file_name} mesh file...")
grid = gm.Grid2D()
grid.read_grid_and_data(file_name)
print(f"Reading {file_name} mesh file is done!")
grid.explicit_solve()
grid.write()

# func.init.p = func.coord_deg_sum(1)
# func.init.T = func.coord_deg_sum(2)
# func.init.u = func.coord_mult_deg(1, [1.0, 1.0])
# func.init.v = func.coord_mult_deg(1, [0.0, 0.0])
# grid.init_var(func.init.p, func.init.T, func.init.u, func.init.v)
#
# print("Calculating gradient...")
# grid.calculate_grad(grad_mode, iter_numb)
# print("Gradient calculation is done!")
#
# print("Calculating divergence...")
# grid.calculate_div()
# print("Divergence calculation is done!")
#
# print("Calculating convective divergence...")
# grid.calculate_convective_div(mode=mode)
# print("Convective divergence calculation is done!")
#
# print("Calculating curl...")
# grid.calculate_curl()
# print("Curl calculation is done!")
#
# print("Calculating laplacian")
# grid.calculate_laplacian()
# print("Laplacian calculation is done!")
#
# print("Output results...")
# grid.write()
# print("Outpit is done!")
#
# print("Work done!")
