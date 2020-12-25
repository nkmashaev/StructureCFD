import numpy as np
from numpy import linalg as la

from spatial.interp import linear_interp


def calc_gradient(
    ni: int,
    nj: int,
    p: np.ndarray,
    gradp: np.ndarray,
    cell_volume: np.ndarray,
    cell_center: np.ndarray,
    i_face_center: np.ndarray,
    i_face_vector: np.ndarray,
    j_face_center: np.ndarray,
    j_face_vector: np.ndarray,
):
    rm = np.zeros(2, dtype=float)
    grad = np.copy(gradp)
    gradp[:, :, :] = 0.0
    gradm = np.zeros(2, dtype=float)

    for i in range(ni):
        for j in range(nj - 1):
            rf = i_face_center[i, j, :]
            sf = i_face_vector[i, j, :]

            r_right = cell_center[i + 1, j + 1, :]
            vol_right = cell_volume[i + 1, j + 1]
            d_right = la.norm(r_right[:] - rf[:])
            p_right = p[i + 1, j + 1]
            gradr = grad[i + 1, j + 1]

            r_left = cell_center[i, j + 1, :]
            vol_left = cell_volume[i, j + 1]
            d_left = la.norm(r_left[:] - rf[:])
            p_left = p[i, j + 1]
            gradl = grad[i, j + 1]

            pm = linear_interp(d_right, d_left, p_right, p_left)
            rm[:] = linear_interp(d_right, d_left, r_right[:], r_left[:])
            gradm[:] = linear_interp(d_right, d_left, gradr[:], gradl[:])
            pf = pm + np.dot(rf[:] - rm[:], gradm[:])

            if vol_left >= 1e-14:
                gradp[i, j + 1, :] += sf[:] * pf / vol_left

            if vol_right >= 1e-14:
                gradp[i + 1, j + 1, :] -= sf[:] * pf / vol_right

    for i in range(ni - 1):
        for j in range(nj):
            rf = j_face_center[i, j, :]
            sf = j_face_vector[i, j, :]

            r_right = cell_center[i + 1, j + 1, :]
            vol_right = cell_volume[i + 1, j + 1]
            d_right = la.norm(r_right[:] - rf[:])
            p_right = p[i + 1, j + 1]
            gradr = grad[i + 1, j + 1]

            r_left = cell_center[i + 1, j, :]
            vol_left = cell_volume[i + 1, j]
            d_left = la.norm(r_left[:] - rf[:])
            p_left = p[i + 1, j]
            gradl = grad[i + 1, j]

            pm = linear_interp(d_right, d_left, p_right, p_left)
            rm[:] = linear_interp(d_right, d_left, r_right[:], r_left[:])
            gradm[:] = linear_interp(d_right, d_left, gradr[:], gradl[:])
            pf = pm + np.dot(rf[:] - rm[:], gradm[:])

            if vol_left >= 1e-14:
                gradp[i + 1, j, :] += sf[:] * pf / vol_left

            if vol_right >= 1e-14:
                gradp[i + 1, j + 1, :] -= sf[:] * pf / vol_right
