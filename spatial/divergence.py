import numpy as np
from numpy import linalg as la

from spatial.interp import linear_interp


def calc_divergence(
    ni: int,
    nj: int,
    v: np.ndarray,
    divv: np.ndarray,
    p: np.ndarray,
    gradp: np.ndarray,
    cell_volume: np.ndarray,
    cell_center: np.ndarray,
    i_face_center: np.ndarray,
    i_face_vector: np.ndarray,
    j_face_center: np.ndarray,
    j_face_vector: np.ndarray,
    mode=0,
):
    divv[:, :] = 0.0
    v_left = np.zeros(2, dtype=float)
    v_right = np.zeros(2, dtype=float)
    vf = np.zeros(2, dtype=float)
    grad_r = np.zeros(2, dtype=float)
    grad_l = np.zeros(2, dtype=float)
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
            if mode == 0:
                pf = 1.0
            else:
                p_right = p[i + 1, j + 1]
                p_left = p[i, j + 1]
                if mode == 1:
                    pf = linear_interp(d_right, d_left, p_right, p_left)
                if mode == 2:
                    if np.dot(sf[:], vf[:]) > 0:
                        if vol_left >= 1e-14:
                            pf = p_left
                        else:
                            pf = 2 * p_left - p_right
                    else:
                        if vol_right >= 1e-14:
                            pf = p_right
                        else:
                            pf = 2 * p_right - p_left
                if mode == 3:
                    grad_r[:] = gradp[i + 1, j + 1, :]
                    grad_l[:] = gradp[i, j + 1, :]
                    if np.dot(sf[:], vf[:]) > 0:
                        if vol_left >= 1e-14:
                            pf = p_left + np.dot(rf[:] - r_left[:], grad_l[:])
                        else:
                            pf = 2 * p_left - p_right
                            gc = np.dot(grad_r[:], r_right[:] - rf[:])
                            gb = p_right - p_left
                            pf += 4.0 * gb - 3.0 * gc
                    else:
                        if vol_right >= 1e-14:
                            pf = p_right + np.dot(rf[:] - r_right[:], grad_r[:])
                        else:
                            pf = 2 * p_right - p_left
                            gc = np.dot(grad_l[:], r_left[:] - rf[:])
                            gb = p_left - p_right
                            pf += 4.0 * gb - 3.0 * gc
            if vol_left >= 1e-14:
                divv[i, j + 1] += pf * np.dot(sf[:], vf[:]) / vol_left
            if vol_right >= 1e-14:
                divv[i + 1, j + 1] -= pf * np.dot(sf[:], vf[:]) / vol_right

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
            if mode == 0:
                pf = 1.0
            else:
                p_right = p[i + 1, j + 1]
                p_left = p[i + 1, j]
                if mode == 1:
                    pf = linear_interp(d_right, d_left, p_right, p_left)
                if mode == 2:
                    if np.dot(sf[:], vf[:]) > 0:
                        if vol_left >= 1e-14:
                            pf = p_left
                        else:
                            pf = 2 * p_left - p_right
                    else:
                        if vol_right >= 1e-14:
                            pf = p_right
                        else:
                            pf = 2 * p_right - p_left
                if mode == 3:
                    grad_r[:] = gradp[i + 1, j + 1, :]
                    grad_l[:] = gradp[i + 1, j, :]
                    if np.dot(sf[:], vf[:]) > 0:
                        if vol_left >= 1e-14:
                            pf = p_left + np.dot(rf[:] - r_left[:], grad_l[:])
                        else:
                            pf = 2 * p_left - p_right
                            gc = np.dot(grad_r[:], r_right[:] - rf[:])
                            gb = p_right - p_left
                            pf += 4.0 * gb - 3.0 * gc
                    else:
                        if vol_right >= 1e-14:
                            pf = p_right + np.dot(rf[:] - r_right[:], grad_r[:])
                        else:
                            pf = 2 * p_right - p_left
                            gc = np.dot(grad_l[:], r_left[:] - rf[:])
                            gb = p_left - p_right
                            pf += 4.0 * gb - 3.0 * gc
            if abs(vol_left) >= 1e-14:
                divv[i + 1, j] += pf * np.dot(sf[:], vf[:]) / vol_left
            if abs(vol_right) >= 1e-14:
                divv[i + 1, j + 1] -= pf * np.dot(sf[:], vf[:]) / vol_right
