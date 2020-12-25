import numpy as np
from numpy import linalg as la

from spatial.interp import linear_interp


def calc_rotor(
    ni: int,
    nj: int,
    v: np.ndarray,
    rot: np.ndarray,
    cell_volume: np.ndarray,
    cell_center: np.ndarray,
    i_face_center: np.ndarray,
    i_face_vector: np.ndarray,
    j_face_center: np.ndarray,
    j_face_vector: np.ndarray,
):
    rot[:, :] = 0.0
    v_left = np.zeros(2, dtype=float)
    v_right = np.zeros(2, dtype=float)
    vf = np.zeros(2, dtype=float)
    for i in range(ni):
        for j in range(nj - 1):
            rf = i_face_center[i, j, :]
            sf = i_face_vector[i, j, :]

            r_right = cell_center[i + 1, j + 1, :]
            vol_right = cell_volume[i + 1, j + 1]
            d_right = la.norm(r_right[:] - rf[:])
            v_right[:] = v[i + 1, j + 1, :]

            r_left = cell_center[i, j + 1, :]
            vol_left = cell_volume[i, j + 1]
            d_left = la.norm(r_left[:] - rf[:])
            v_left[:] = v[i, j + 1, :]

            vf[:] = linear_interp(d_right, d_left, v_right[:], v_left[:])
            add = -(sf[0] * vf[1] - sf[1] * vf[0])
            if abs(vol_left) >= 1e-14:
                rot[i, j + 1] += add / vol_left

            if abs(vol_right) >= 1e-14:
                rot[i + 1, j + 1] -= add / vol_right

    for i in range(ni - 1):
        for j in range(nj):
            rf = j_face_center[i, j, :]
            sf = j_face_vector[i, j, :]

            r_right = cell_center[i + 1, j + 1, :]
            vol_right = cell_volume[i + 1, j + 1]
            d_right = la.norm(r_right[:] - rf[:])
            v_right[:] = v[i + 1, j + 1, :]

            r_left = cell_center[i + 1, j, :]
            vol_left = cell_volume[i + 1, j]
            d_left = la.norm(r_left[:] - rf[:])
            v_left[:] = v[i + 1, j, :]

            vf[:] = linear_interp(d_right, d_left, v_right[:], v_left[:])
            add = -(sf[0] * vf[1] - sf[1] * vf[0])
            if abs(vol_left) >= 1e-14:
                rot[i + 1, j] += add / vol_left
            if abs(vol_right) >= 1e-14:
                rot[i + 1, j + 1] -= add / vol_right
