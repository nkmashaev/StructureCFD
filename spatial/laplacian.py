import numpy as np
from numpy import linalg as la

from spatial.interp import linear_interp


def calc_laplacian(
    ni: int,
    nj: int,
    p: np.ndarray,
    gradp: np.ndarray,
    laplacian: np.ndarray,
    cell_volume: np.ndarray,
    cell_center: np.ndarray,
    i_face_center: np.ndarray,
    i_face_vector: np.ndarray,
    j_face_center: np.ndarray,
    j_face_vector: np.ndarray,
):
    laplacian[:, :] = 0.0
    grad_l = np.zeros(2, dtype=float)
    grad_r = np.zeros(2, dtype=float)
    gradf = np.zeros(2, dtype=float)
    rnc = np.zeros(2, dtype=float)
    for i in range(ni):
        for j in range(nj - 1):
            rf = i_face_center[i, j, :]
            sf = i_face_vector[i, j, :]
            nf = i_face_vector[i, j, :] / la.norm(sf[:])

            r_right = cell_center[i + 1, j + 1, :]
            vol_right = cell_volume[i + 1, j + 1]
            d_right = la.norm(r_right[:] - rf[:])
            p_right = p[i + 1, j + 1]
            grad_r[:] = gradp[i + 1, j + 1, :]

            r_left = cell_center[i, j + 1, :]
            vol_left = cell_volume[i, j + 1]
            d_left = la.norm(r_left[:] - rf[:])
            p_left = p[i, j + 1]
            grad_l[:] = gradp[i, j + 1, :]

            dnc = la.norm(r_right[:] - r_left[:])
            rnc[:] = (r_right[:] - r_left[:]) / dnc

            gradf[:] = linear_interp(d_right, d_left, grad_r[:], grad_l[:])
            dpdn = (p_right - p_left) / dnc

            if abs(vol_left) < 1e-14:
                dpdn_c = np.dot(grad_r[:], nf[:])
                dpdn = (5.0 * dpdn - 2.0 * dpdn_c) / 3.0
                gradf[:] = grad_r[:]
            if abs(vol_right) < 1e-14:
                dpdn_c = np.dot(grad_l[:], nf[:])
                dpdn = (5.0 * dpdn - 2.0 * dpdn_c) / 3.0
                gradf[:] = grad_l[:]
            dpdn += np.dot(nf[:] - rnc[:], gradf[:])

            if abs(vol_left) >= 1e-14:
                laplacian[i, j + 1] += dpdn * la.norm(sf) / vol_left

            if abs(vol_right) >= 1e-14:
                laplacian[i + 1, j + 1] -= dpdn * la.norm(sf) / vol_right

    for i in range(ni - 1):
        for j in range(nj):
            rf = j_face_center[i, j, :]
            sf = j_face_vector[i, j, :]
            nf = j_face_vector[i, j, :] / la.norm(sf[:])

            r_right = cell_center[i + 1, j + 1, :]
            vol_right = cell_volume[i + 1, j + 1]
            d_right = la.norm(r_right[:] - rf[:])
            p_right = p[i + 1, j + 1]
            grad_r[:] = gradp[i + 1, j + 1, :]

            r_left = cell_center[i + 1, j, :]
            vol_left = cell_volume[i + 1, j]
            d_left = la.norm(r_left[:] - rf[:])
            p_left = p[i + 1, j]
            grad_l[:] = gradp[i + 1, j, :]

            dnc = la.norm(r_right[:] - r_left[:])
            rnc[:] = (r_right[:] - r_left[:]) / dnc

            gradf[:] = linear_interp(d_right, d_left, grad_r[:], grad_l[:])
            dpdn = (p_right - p_left) / dnc

            if abs(vol_left) < 1e-14:
                dpdn_c = np.dot(grad_r[:], nf[:])
                dpdn = (5.0 * dpdn - 2.0 * dpdn_c) / 3.0
                gradf[:] = grad_r[:]
            if abs(vol_right) < 1e-14:
                dpdn_c = np.dot(grad_l[:], nf[:])
                dpdn = (5.0 * dpdn - 2.0 * dpdn_c) / 3.0
                gradf[:] = grad_l[:]
            dpdn += np.dot(nf[:] - rnc[:], gradf[:])

            if abs(vol_left) >= 1e-14:
                laplacian[i + 1, j] += dpdn * la.norm(sf) / vol_left

            if abs(vol_right) >= 1e-14:
                laplacian[i + 1, j + 1] -= dpdn * la.norm(sf) / vol_right
