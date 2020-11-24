import functools
import re
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

import cfd.functions as func


def with_iteration(numb: int) -> Callable[[Any], Any]:
    def decorator(func: Callable[[Any], Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(numb):
                func(*args, **kwargs)

        return wrapper

    return decorator


class OutputManager:
    """
    Class OutputManager is designed to output cell centered data
    into Tecplot file
    """

    __slots__ = ("__nodes", "__output_dict", "__i_size", "__j_size")

    def __init__(self, nodes: np.ndarray):
        """
        OutputManager constructor

        :param nodes: array of grid nodes
        """
        self.__nodes = nodes
        self.__i_size, self.__j_size, *_ = nodes.shape
        self.__output_dict = {}

    @property
    def header(self) -> str:
        """
        Return names of stored variables in the following format:

        Variables="X", "Y", ...

        :return: stored variable names string
        """
        header = 'Variables="X", "Y"'
        for key in self.__output_dict.keys():
            header += f', "{key}"'
        return header

    @property
    def storage_info(self) -> str:
        """
        Return information about stored arrays dimension, data packing type
        and var location

        :return: storage info string
        """
        param_numb = len(self.__output_dict) + 2
        storage_info = f"Zone i={self.__i_size}, j={self.__j_size}, "
        storage_info += (
            f"DATAPACKING=BLOCK, VARLOCATION=([3-{param_numb}]=CELLCENTERED)"
        )
        return storage_info

    def save_scalar(self, scalar_name: str, scalar_data: np.ndarray):
        """
        Accepts variable name and field and marked it for output.
        Note that variable should be scalar.

        :param scalar_name: name of variable
        :param scalar_data: variable 2D field
        """
        if len(scalar_data.shape) > 2:
            raise TypeError("Error: Expected scalar value!")
        i_size, j_size = scalar_data.shape
        if i_size != self.__i_size + 1 or j_size != self.__j_size + 1:
            raise AttributeError("Error: Dimension mismatch!")

        self.__output_dict[scalar_name] = scalar_data

    def output(self, file_path: str):
        """
        Output marked with save_scalar method data to file in Tecplot format.

        :param file_path: name of file
        """
        with open(file_path, "w") as out_file:
            i = self.__i_size
            j = self.__j_size

            out_file.write(self.header + "\n")
            out_file.write(self.storage_info + "\n")
            np.savetxt(
                out_file,
                self.__nodes[:, :, 0].reshape(1, i * j),
                fmt="%.11e",
                delimiter="\n",
            )
            np.savetxt(
                out_file,
                self.__nodes[:, :, 1].reshape(1, i * j),
                fmt="%.11e",
                delimiter="\n",
            )
            for val in self.__output_dict.values():
                np.savetxt(
                    out_file,
                    val[1:i, 1:j].reshape(1, (i - 1) * (j - 1)),
                    fmt="%.11e",
                    delimiter="\n",
                )


class Grid2D:
    """
    Class Grid2D is designed to work with two-dimensional
    finite volume cell-centered grids
    """

    __slots__ = (
        "__nodes",
        "__ni",
        "__nj",
        "__i_face_vector",
        "__i_face_center",
        "__j_face_vector",
        "__j_face_center",
        "__cell_volume",
        "__cell_center",
        "__p",
        "__grad_p",
        "__laplacian_p",
        "__V",
        "__div_V",
        "__curl_Vz",
        "__grad_u",
        "__grad_v",
        "__div_Vu",
        "__div_Vv",
        "__laplacian_u",
        "__laplacian_v",
        "__T",
        "__grad_T",
        "__laplacian_T",
        "__div_pV",
        "__div_TV",
        "__theta",
        "__dc",
        "__dn",
        "__mf",
        "__dnc",
        "__nf_rnc",
        "__nf",
        "__rnf",
        "__rcf",
    )

    def __init__(self):
        self.__nodes = None
        self.__i_face_vector = None
        self.__i_face_center = None
        self.__j_face_vector = None
        self.__j_face_center = None
        self.__cell_volume = None
        self.__cell_center = None
        self.__p = None
        self.__grad_p = None
        self.__laplacian_p = None
        self.__T = None
        self.__grad_T = None
        self.__laplacian_T = None
        self.__V = None
        self.__grad_u = None
        self.__grad_v = None
        self.__div_V = None
        self.__curl_Vz = None
        self.__div_pV = None
        self.__div_TV = None
        self.__div_Vu = None
        self.__div_Vv = None
        self.__laplacian_u = None
        self.__laplacian_v = None
        self.__theta = None
        self.__dc = None
        self.__dn = None
        self.__dnc = None
        self.__mf = None
        self.__nf_rnc = None
        self.__rnf = None
        self.__rcf = None
        self.__nf = None
        self.__ni = 0
        self.__nj = 0

    def write(self):
        xn = self.__nodes
        out = OutputManager(xn)
        out.save_scalar("Pressure", self.__p)
        out.save_scalar("X-GradientPressure", self.__grad_p[:, :, 0])
        out.save_scalar("Y-GradientPressure", self.__grad_p[:, :, 1])
        out.save_scalar("Pressure Convective Div", self.__div_pV)
        out.save_scalar("Laplacian Pressure", self.__laplacian_p)
        out.save_scalar("Temperature", self.__T)
        out.save_scalar("X-GradientTemperature", self.__grad_T[:, :, 0])
        out.save_scalar("Y-GradientTemperature", self.__grad_T[:, :, 1])
        out.save_scalar("Temperature Convective Div", self.__div_TV)
        out.save_scalar("Laplacian Temperature", self.__laplacian_T)
        out.save_scalar("X-Velocity", self.__V[:, :, 0])
        out.save_scalar("Y-Velocity", self.__V[:, :, 1])
        out.save_scalar("X-GradientU", self.__grad_u[:, :, 0])
        out.save_scalar("Y-GradientU", self.__grad_u[:, :, 1])
        out.save_scalar("X-GradientV", self.__grad_v[:, :, 0])
        out.save_scalar("Y-GradientV", self.__grad_v[:, :, 1])
        out.save_scalar("Velocity Divergence", self.__div_V)
        out.save_scalar("U Convective Div", self.__div_Vu[:, :])
        out.save_scalar("V Convective Div", self.__div_Vv[:, :])
        out.save_scalar("U Laplacian", self.__laplacian_u[:, :])
        out.save_scalar("V Laplacian", self.__laplacian_v[:, :])
        out.save_scalar("Z-Curl Velocity", self.__curl_Vz)
        out.output("data.plt")

        plt.title("Temperature")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.pcolor(
            self.__nodes[:, :, 0],
            self.__nodes[:, :, 1],
            self.__T[1:-1, 1:-1],
            edgecolor="k",
            linewidth=1,
            cmap="rainbow",
        )
        plt.colorbar()
        plt.scatter(
            self.__cell_center[1:-1, 1:-1, 0],
            self.__cell_center[1:-1, 1:-1, 1],
            s=0.2,
            c="k",
        )
        plt.show()

    def __neighbour_cells(self, i: int, j: int) -> np.ndarray:
        ncell = np.array(
            [
                [i - 1, j],
                [i + 1, j],
                [i, j - 1],
                [i, j + 1],
            ],
            dtype=int,
        )
        return ncell

    def __rf(self, i: int, j: int) -> np.ndarray:
        rf = np.array(
            [
                self.__i_face_center[i - 1, j - 1, :],
                self.__i_face_center[i, j - 1, :],
                self.__j_face_center[i - 1, j - 1, :],
                self.__j_face_center[i - 1, j, :],
            ]
        )
        return rf

    def __sf(self, i: int, j: int) -> np.ndarray:
        sf = np.array(
            [
                self.__i_face_vector[i - 1, j - 1, :],
                -self.__i_face_vector[i, j - 1, :],
                self.__j_face_vector[i - 1, j - 1, :],
                -self.__j_face_vector[i - 1, j, :],
            ]
        )
        return sf

    def __init_field_operator(self):

        self.__theta = np.zeros((self.__ni + 1, self.__nj + 1, 4, 2), dtype=float)
        self.__dc = np.zeros((self.__ni + 1, self.__nj + 1, 4), dtype=float)
        self.__dn = np.zeros((self.__ni + 1, self.__nj + 1, 4), dtype=float)
        self.__dnc = np.zeros((self.__ni + 1, self.__nj + 1, 4), dtype=float)
        self.__mf = np.zeros((self.__ni + 1, self.__nj + 1, 4, 2), dtype=float)
        self.__nf_rnc = np.zeros((self.__ni + 1, self.__nj + 1, 4, 2), dtype=float)
        self.__nf = np.zeros((self.__ni + 1, self.__nj + 1, 4, 2), dtype=float)
        self.__rnf = np.zeros((self.__ni + 1, self.__nj + 1, 4, 2), dtype=float)
        self.__rcf = np.zeros((self.__ni + 1, self.__nj + 1, 4, 2), dtype=float)
        for i in range(1, self.__ni):
            for j in range(1, self.__nj):
                ncell = self.__neighbour_cells(i, j)
                sf = self.__sf(i, j)
                rc = self.__cell_center[i, j, :]
                rf = self.__rf(i, j)
                dx_dx = 0.0
                dx_dy = 0.0
                dy_dy = 0.0
                for iface, neighbour in enumerate(ncell):
                    i_n, j_n = neighbour
                    rn = self.__cell_center[i_n, j_n, :]
                    dx = rn[0] - rc[0]
                    dy = rn[1] - rc[1]
                    self.__dc[i, j, iface] = LA.norm(rf[iface, :] - rc[:])
                    self.__dn[i, j, iface] = LA.norm(rf[iface, :] - rn[:])
                    self.__rnf[i, j, iface, :] = rf[iface, :] - rn[:]
                    self.__rcf[i, j, iface, :] = rf[iface, :] - rc[:]
                    rm = np.array(
                        [
                            func.linear_interpolation(
                                self.__dc[i, j, iface],
                                self.__dn[i, j, iface],
                                rc[k],
                                rn[k],
                            )
                            for k in range(2)
                        ],
                        dtype=float,
                    )
                    self.__mf[i, j, iface, :] = rf[iface, :] - rm[:]
                    self.__nf[i, j, iface, :] = sf[iface, :] / LA.norm(sf[iface, :])
                    self.__dnc[i, j, iface] = LA.norm(rn[:] - rc[:])
                    self.__nf_rnc[i, j, iface, :] = (
                        self.__nf[i, j, iface, :]
                        - (rn[:] - rc[:]) / self.__dnc[i, j, iface]
                    )
                    weight = 1 / (dx * dx + dy * dy)
                    dx_dx += dx * dx * weight
                    dx_dy += dx * dy * weight
                    dy_dy += dy * dy * weight
                r11 = np.sqrt(dx_dx)
                r12 = dx_dy / r11
                r22 = np.sqrt(dy_dy - r12 * r12)

                for iface, neighbour in enumerate(ncell):
                    i_n, j_n = neighbour
                    rn = self.__cell_center[i_n, j_n, :]
                    dx = rn[0] - rc[0]
                    dy = rn[1] - rc[1]
                    weight = 1 / (dx * dx + dy * dy)
                    a1 = dx / (r11 * r11)
                    a2 = (dy - r12 * dx / r11) / (r22 * r22)
                    self.__theta[i, j, iface, :] = (
                        np.array([a1 - r12 * a2 / r11, a2]) * weight
                    )

    def __green_gauss_scalar(self, i: int, j: int, var: np.ndarray, grad: np.ndarray):

        ncell = self.__neighbour_cells(i, j)
        sf = self.__sf(i, j)
        vol = self.__cell_volume[i - 1, j - 1]
        gradc = np.zeros(2)
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            xm = func.linear_interpolation(
                self.__dc[i, j, iface], self.__dn[i, j, iface], var[i, j], var[i_n, j_n]
            )
            gradm = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grad[i, j, k],
                        grad[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            xf = xm + np.dot(self.__mf[i, j, iface, :], gradm[:])
            gradc[:] += xf * sf[iface, :]
        grad[i, j, :] = gradc[:] / vol

    def __green_gauss_vector(
        self, i: int, j: int, var: np.ndarray, gradx: np.ndarray, grady: np.ndarray
    ):

        ncell = self.__neighbour_cells(i, j)
        sf = self.__sf(i, j)
        vol = self.__cell_volume[i - 1, j - 1]
        gradcx = np.zeros(2)
        gradcy = np.zeros(2)
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            xm = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        var[i, j, k],
                        var[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )

            gradmx = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        gradx[i, j, k],
                        gradx[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )

            gradmy = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grady[i, j, k],
                        grady[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )

            xfx = xm[0] + np.dot(self.__mf[i, j, iface, :], gradmx[:])
            xfy = xm[1] + np.dot(self.__mf[i, j, iface, :], gradmy[:])
            gradcx[:] += xfx * sf[iface, :]
            gradcy[:] += xfy * sf[iface, :]
        gradx[i, j, :] = gradcx[:] / vol
        grady[i, j, :] = gradcy[:] / vol

    def __least_squares_scalar(self, i: int, j: int, var: np.ndarray, grad: np.ndarray):
        ncell = self.__neighbour_cells(i, j)
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            grad[i, j, :] += self.__theta[i, j, iface, :] * (var[i_n, j_n] - var[i, j])

    def __least_squares_vector(
        self, i: int, j: int, var: np.ndarray, gradx: np.ndarray, grady: np.ndarray
    ):
        ncell = self.__neighbour_cells(i, j)
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            gradx[i, j, :] += self.__theta[i, j, iface, :] * (
                var[i_n, 0] - var[i, j, 0]
            )
            grady[i, j, :] += self.__theta[i, j, iface, :] * (
                var[i_n, 1] - var[i, j, 1]
            )

    def __divergence(
        self,
        i: int,
        j: int,
        var: np.ndarray,
        gradx: np.ndarray,
        grady: np.ndarray,
        div: np.ndarray,
    ):
        ncell = self.__neighbour_cells(i, j)
        sf = self.__sf(i, j)
        vol = self.__cell_volume[i - 1, j - 1]
        xf = np.zeros(2)
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            xm = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        var[i, j, k],
                        var[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradm_x = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        gradx[i, j, k],
                        gradx[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradm_y = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grady[i, j, k],
                        grady[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            xf[0] = xm[0] + np.dot(self.__mf[i, j, iface, :], gradm_x[:])
            xf[1] = xm[1] + np.dot(self.__mf[i, j, iface, :], gradm_y[:])
            div[i, j] += np.dot(xf[:], sf[iface, :])
        div[i, j] = div[i, j] / vol

    def __conv_div_scalar(
        self,
        i: int,
        j: int,
        var: np.ndarray,
        grad: np.ndarray,
        cdiv: np.ndarray,
        mode: int,
    ):
        vel = self.__V
        grad_u = self.__grad_u
        grad_v = self.__grad_v
        ncell = self.__neighbour_cells(i, j)
        sf = self.__sf(i, j)
        vol = self.__cell_volume[i - 1, j - 1]
        velf = np.zeros(2)
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            velm = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        vel[i, j, k],
                        vel[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradm_u = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grad_u[i, j, k],
                        grad_u[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradm_v = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grad_v[i, j, k],
                        grad_v[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            velf[0] = velm[0] + np.dot(self.__mf[i, j, iface, :], gradm_u[:])
            velf[1] = velm[1] + np.dot(self.__mf[i, j, iface, :], gradm_v[:])
            if mode == 1:
                xf = func.linear_interpolation(
                    self.__dc[i, j, iface],
                    self.__dn[i, j, iface],
                    var[i, j],
                    var[i_n, j_n],
                )
                cdiv[i, j] += np.dot(xf * velf[:], sf[iface, :])
            elif mode == 2:
                if np.dot(velf[:], sf[iface, :]) >= 0:
                    xf = var[i, j]
                else:
                    if self.is_boundary(i_n, j_n):
                        xf = 2.0 * var[i_n, j_n] - var[i, j]
                    else:
                        xf = var[i_n, j_n]
                cdiv[i, j] += np.dot(xf * velf[:], sf[iface, :])
            elif mode == 3:
                if np.dot(velf[:], sf[iface, :]) >= 0:
                    xf = var[i, j] + np.dot(grad[i, j, :], self.__rcf[i, j, iface, :])
                else:
                    if self.is_boundary(i_n, j_n):
                        xf_fo = 2.0 * var[i_n, j_n] - var[i, j]
                        grad_c_dot_r = -np.dot(
                            grad[i, j, :], self.__rcf[i, j, iface, :]
                        )
                        grad_b_dot_r = var[i, j] - var[i_n, j_n]
                        grad_d_dot_r = 4.0 * grad_b_dot_r - 3.0 * grad_c_dot_r
                        xf = xf_fo + grad_d_dot_r
                    else:
                        xf = var[i_n, j_n] + np.dot(
                            grad[i_n, j_n, :], self.__rnf[i, j, iface, :]
                        )

                cdiv[i, j] += np.dot(xf * velf, sf[iface, :])
        cdiv[i, j] /= vol

    def __conv_div_vector(
        self,
        i: int,
        j: int,
        var: np.ndarray,
        gradx: np.ndarray,
        grady: np.ndarray,
        cdivx: np.ndarray,
        cdivy: np.ndarray,
        mode: int,
    ):
        vel = self.__V
        grad_u = self.__grad_u
        grad_v = self.__grad_v
        ncell = self.__neighbour_cells(i, j)
        sf = self.__sf(i, j)
        vol = self.__cell_volume[i - 1, j - 1]
        velf = np.zeros(2)
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            velm = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        vel[i, j, k],
                        vel[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradm_u = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grad_u[i, j, k],
                        grad_u[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradm_v = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grad_v[i, j, k],
                        grad_v[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            velf[0] = velm[0] + np.dot(self.__mf[i, j, iface, :], gradm_u[:])
            velf[1] = velm[1] + np.dot(self.__mf[i, j, iface, :], gradm_v[:])
            if mode == 1:
                xf = np.array(
                    [
                        func.linear_interpolation(
                            self.__dc[i, j, iface],
                            self.__dn[i, j, iface],
                            var[i, j, k],
                            var[i_n, j_n, k],
                        )
                        for k in range(2)
                    ],
                    dtype=float,
                )
                cdivx[i, j] += np.dot(xf[0] * velf[:], sf[iface, :])
                cdivy[i, j] += np.dot(xf[1] * velf[:], sf[iface, :])
            elif mode == 2:
                if np.dot(velf[:], sf[iface, :]) >= 0:
                    xf = np.array([var[i, j, k] for k in range(2)], dtype=float)
                else:
                    if self.is_boundary(i_n, j_n):
                        xf = np.array(
                            [2.0 * var[i_n, j_n, k] - var[i, j, k] for k in range(2)],
                            dtype=float,
                        )
                    else:
                        xf = np.array([var[i_n, j_n, k] for k in range(2)], dtype=float)
                cdivx[i, j] += np.dot(xf[0] * velf[:], sf[iface, :])
                cdivy[i, j] += np.dot(xf[1] * velf[:], sf[iface, :])
            elif mode == 3:
                if np.dot(velf[:], sf[iface, :]) >= 0:
                    xf = np.zeros(2, dtype=float)
                    xf[0] = var[i, j, 0] + np.dot(
                        gradx[i, j, :], self.__rcf[i, j, iface, :]
                    )
                    xf[1] = var[i, j, 1] + np.dot(
                        grady[i, j, :], self.__rcf[i, j, iface, :]
                    )
                else:
                    if self.is_boundary(i_n, j_n):
                        xf_fo = np.array(
                            [2.0 * var[i_n, j_n, k] - var[i, j, k] for k in range(2)],
                            dtype=float,
                        )
                        grad_c_dot_r = np.zeros(2, dtype=float)
                        grad_c_dot_r[0] = -np.dot(
                            gradx[i, j, :], self.__rcf[i, j, iface, :]
                        )
                        grad_c_dot_r[1] = -np.dot(
                            grady[i, j, :], self.__rcf[i, j, iface, :]
                        )
                        grad_b_dot_r = np.array(
                            [var[i, j, k] - var[i_n, j_n, k] for k in range(2)],
                            dtype=float,
                        )
                        grad_d_dot_r = 4.0 * grad_b_dot_r - 3.0 * grad_c_dot_r
                        xf = xf_fo[:] + grad_d_dot_r[:]
                    else:
                        xf = np.zeros(2, dtype=float)
                        xf[0] = var[i_n, j_n, 0] + np.dot(
                            gradx[i_n, j_n, :], self.__rnf[i, j, iface, :]
                        )
                        xf[1] = var[i_n, j_n, 1] + np.dot(
                            grady[i_n, j_n, :], self.__rnf[i, j, iface, :]
                        )

                cdivx[i, j] += np.dot(xf[0] * velf, sf[iface, :])
                cdivy[i, j] += np.dot(xf[1] * velf, sf[iface, :])
        cdivx[i, j] /= vol
        cdivy[i, j] /= vol

    def __curl(
        self,
        i: int,
        j: int,
        var: np.ndarray,
        gradx: np.ndarray,
        grady: np.ndarray,
        curl_z: np.ndarray,
    ):
        ncell = self.__neighbour_cells(i, j)
        sf = self.__sf(i, j)
        vol = self.__cell_volume[i - 1, j - 1]
        xf = np.zeros(2)
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            xm = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        var[i, j, k],
                        var[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradm_x = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        gradx[i, j, k],
                        gradx[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradm_y = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grady[i, j, k],
                        grady[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            xf[0] = xm[0] + np.dot(self.__mf[i, j, iface, :], gradm_x[:])
            xf[1] = xm[1] + np.dot(self.__mf[i, j, iface, :], gradm_y[:])
            curl_z[i, j] += np.cross(xf[:], sf[iface, :])
        curl_z[i, j] = curl_z[i, j] / vol

    def __laplacian_scalar(
        self, i: int, j: int, var: np.ndarray, grad: np.ndarray, laplacian: np.ndarray
    ):
        ncell = self.__neighbour_cells(i, j)
        sf = self.__sf(i, j)
        vol = self.__cell_volume[i - 1, j - 1]
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            gradf = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grad[i, j, k],
                        grad[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            dxdn = (var[i_n, j_n] - var[i, j]) / self.__dnc[i, j, iface]

            if self.is_boundary(i_n, j_n):
                dxdn_c = np.dot(grad[i, j, :], self.__nf[i, j, iface, :])
                dxdn = (5.0 * dxdn - 2.0 * dxdn_c) / 3.0
                gradf[:] = grad[i, j, :]

            dxdn += np.dot(self.__nf_rnc[i, j, iface, :], gradf[:])
            laplacian[i, j] += dxdn * LA.norm(sf[iface, :])
        laplacian[i, j] /= vol

    def __laplacian_vector(
        self,
        i: int,
        j: int,
        var: np.ndarray,
        gradx: np.ndarray,
        grady: np.ndarray,
        laplacianx: np.ndarray,
        laplaciany: np.ndarray,
    ):
        ncell = self.__neighbour_cells(i, j)
        sf = self.__sf(i, j)
        vol = self.__cell_volume[i - 1, j - 1]
        for iface, neighbour in enumerate(ncell):
            i_n, j_n = neighbour
            gradfx = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        gradx[i, j, k],
                        gradx[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            gradfy = np.array(
                [
                    func.linear_interpolation(
                        self.__dc[i, j, iface],
                        self.__dn[i, j, iface],
                        grady[i, j, k],
                        grady[i_n, j_n, k],
                    )
                    for k in range(2)
                ],
                dtype=float,
            )
            dxdn = np.array(
                [
                    (var[i_n, j_n, k] - var[i, j, k]) / self.__dnc[i, j, iface]
                    for k in range(2)
                ],
                dtype=float,
            )

            if self.is_boundary(i_n, j_n):
                dxdn_c = np.zeros(2)
                dxdn_c[0] = np.dot(gradx[i, j, :], self.__nf[i, j, iface, :])
                dxdn_c[1] = np.dot(grady[i, j, :], self.__nf[i, j, iface, :])
                dxdn[:] = (5.0 * dxdn[:] - 2.0 * dxdn_c[:]) / 3.0
                gradfx[:] = gradx[i, j, :]
                gradfy[:] = grady[i, j, :]

            dxdn[0] += np.dot(self.__nf_rnc[i, j, iface, :], gradfx[:])
            dxdn[1] += np.dot(self.__nf_rnc[i, j, iface, :], gradfy[:])
            laplacianx[i, j] += dxdn[0] * LA.norm(sf[iface, :])
            laplaciany[i, j] += dxdn[1] * LA.norm(sf[iface, :])
        laplacianx[i, j] /= vol
        laplaciany[i, j] /= vol

    def __calc_metric(self):
        """
        Determine faces and cells: calculate faces' squares and centers,
        compute cells' volumes and centers, determine boundaries, creating
        normal vectors
        """
        xn = self.__nodes
        ni = self.__ni
        nj = self.__nj
        x = 0
        y = 1

        # i - direction faces initialization
        # i_face_vector - normal vectors
        # i_face_center - faces centers
        i_face_vector = np.zeros((ni, nj - 1, 2), dtype=float)
        i_face_center = np.zeros((ni, nj - 1, 2), dtype=float)
        for i in range(ni):
            for j in range(nj - 1):
                i_face_vector[i, j, x] = xn[i, j + 1, y] - xn[i, j, y]
                i_face_vector[i, j, y] = -(xn[i, j + 1, x] - xn[i, j, x])
                i_face_center[i, j, :] = 0.5 * (xn[i, j + 1, :] + xn[i, j, :])
        self.__i_face_vector = i_face_vector
        self.__i_face_center = i_face_center

        # j - direction faces initialization
        # j_face_vector - normal vectors
        # j_face_center - faces centers
        j_face_vector = np.zeros((ni - 1, nj, 2), dtype=float)
        j_face_center = np.zeros((ni - 1, nj, 2), dtype=float)
        for i in range(ni - 1):
            for j in range(nj):
                j_face_vector[i, j, 0] = -(xn[i + 1, j, y] - xn[i, j, y])
                j_face_vector[i, j, 1] = xn[i + 1, j, x] - xn[i, j, x]
                j_face_center[i, j, :] = 0.5 * (xn[i + 1, j, :] + xn[i, j, :])
        self.__j_face_vector = j_face_vector
        self.__j_face_center = j_face_center

        # cell volumes storage
        # volume calculates by sum of two triangles square
        # triangle square calculated by dot product
        cell_volume = np.zeros((ni - 1, nj - 1), dtype=float)
        rv = np.zeros(2)
        for i in range(ni - 1):
            for j in range(nj - 1):
                rv[x] = xn[i + 1, j + 1, x] - xn[i, j, x]
                rv[y] = xn[i + 1, j + 1, y] - xn[i, j, y]
                a = i_face_vector[i, j, :]
                b = j_face_vector[i, j, :]
                cell_volume[i, j] = 0.5 * (
                    np.abs(np.dot(a, rv)) + np.abs(np.dot(b, rv))
                )
        self.__cell_volume = cell_volume

        # cell centers storage
        # calculation for inside volumes
        cell_center = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        faces = [0] * 4
        for i in range(1, ni):
            for j in range(1, nj):
                # faces centers radius vectors
                faces[0] = (
                    i_face_center[i - 1, j - 1, :],
                    i_face_vector[i - 1, j - 1, :],
                )
                faces[1] = (i_face_center[i, j - 1, :], i_face_vector[i, j - 1, :])
                faces[2] = (
                    j_face_center[i - 1, j - 1, :],
                    j_face_vector[i - 1, j - 1, :],
                )
                faces[3] = (j_face_center[i - 1, j, :], j_face_vector[i - 1, j, :])

                total_square = 0.0
                cell_center[i, j, :] = 0.0
                for face, sf in faces:
                    face_square = LA.norm(sf)
                    cell_center[i, j, :] += face[:] * face_square
                    total_square += face_square
                cell_center[i, j, :] /= total_square

        # Calculation for boundary vols
        for j in range(1, nj):
            cell_center[0, j, :] = i_face_center[0, j - 1, :]
            cell_center[-1, j, :] = i_face_center[-1, j - 1, :]
        for i in range(1, ni):
            cell_center[i, 0, :] = j_face_center[i - 1, 0, :]
            cell_center[i, -1, :] = j_face_center[i - 1, -1, :]
        self.__cell_center = cell_center

    def init_var(
        self,
        p_init: Callable[[Any], Any],
        T_init: Callable[[Any], Any],
        u_init: Callable[[Any], Any],
        v_init: Callable[[Any], Any],
    ):
        ni = self.__ni
        nj = self.__nj
        xn = self.__nodes
        xc = self.__cell_center

        self.__grad_p = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__laplacian_p = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__grad_T = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__laplacian_T = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__grad_u = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__grad_v = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__div_V = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__div_Vu = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__div_Vv = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__laplacian_u = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__laplacian_v = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__div_pV = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__div_TV = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__curl_Vz = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__V = np.zeros((ni + 1, nj + 1, 2), dtype=float)

        self.__p = np.array(
            [
                [p_init(xc[i, j, 0], xc[i, j, 1]) for j in range(nj + 1)]
                for i in range(ni + 1)
            ],
            dtype=float,
        )
        self.__T = np.array(
            [
                [T_init(xc[i, j, 0], xc[i, j, 1]) for j in range(nj + 1)]
                for i in range(ni + 1)
            ],
            dtype=float,
        )
        self.__V[:, :, 0] = np.array(
            [
                [u_init(xc[i, j, 0], xc[i, j, 1]) for j in range(nj + 1)]
                for i in range(ni + 1)
            ],
            dtype=float,
        )
        self.__V[:, :, 1] = np.array(
            [
                [v_init(xc[i, j, 0], xc[i, j, 1]) for j in range(nj + 1)]
                for i in range(ni + 1)
            ],
            dtype=float,
        )

    def read_grid(self, mesh_name: str):
        """
        Read structured grid nodes data from file with mesh_name.
        Then calculate parameters of grid

        :param mesh_name: name of grid file
        """
        with open(mesh_name, "r") as in_file:
            # read grid dimension
            self.__ni, self.__nj = map(int, in_file.readline().split())

            # read nodes' coordinates from file with name mesh_name
            nodes = np.loadtxt(fname=mesh_name, usecols=(0, 1), skiprows=1, dtype=float)

            # reshape 1-D input arrays to 2-D
            self.__nodes = nodes.reshape((self.__ni, self.__nj, 2))
        self.__calc_metric()
        self.__init_field_operator()

    def read_grid_and_data(self, file_name: str):
        with open(file_name, "r") as in_file:
            # read grid dimension
            curr_str = ""
            ni_search = None
            while ni_search is None:
                curr_str = in_file.readline()
                ni_search = re.search(r"[i]=(\d+)", curr_str)
            self.__ni = int(ni_search.group(1))

            nj_search = re.search(r"[j]=(\d+)", curr_str)
            while nj_search is None:
                curr_str = in_file.readline()
                nj_search = re.search(r"[j]=(\d+)", curr_str)

            self.__nj = int(nj_search.group(1))

            # self.__ni, self.__nj = map(int, in_file.readline().split())

            # read nodes' coordinates from file with name mesh_name
            x, y, u, v, p = np.loadtxt(
                fname=file_name,
                usecols=(0, 1, 2, 3, 5),
                skiprows=2,
                unpack=True,
                dtype=float,
            )
            x = x.reshape((self.__ni, self.__nj))
            y = y.reshape((self.__ni, self.__nj))
            u = u.reshape((self.__ni, self.__nj))
            v = v.reshape((self.__ni, self.__nj))
            p = p.reshape((self.__ni, self.__nj))
            self.__nodes = np.array(
                [
                    [
                        [x[i, j] if k == 0 else y[i, j] for k in range(2)]
                        for j in range(self.__nj)
                    ]
                    for i in range(self.__ni)
                ]
            )
        self.__calc_metric()
        ni = self.__ni
        nj = self.__nj
        self.__grad_p = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__laplacian_p = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__grad_T = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__laplacian_T = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__grad_u = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__grad_v = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__div_V = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__div_Vu = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__div_Vv = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__laplacian_u = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__laplacian_v = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__div_pV = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__div_TV = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__curl_Vz = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__V = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        self.__p = np.zeros((ni + 1, nj + 1), dtype=float)
        self.__T = np.zeros((ni + 1, nj + 1), dtype=float)

        for i in range(1, ni):
            for j in range(1, nj):
                rc = self.__cell_center[i, j, :]
                d1 = LA.norm(self.__nodes[i - 1, j - 1, :] - rc)
                d2 = LA.norm(self.__nodes[i, j - 1, :] - rc)
                d3 = LA.norm(self.__nodes[i, j, :] - rc)
                d4 = LA.norm(self.__nodes[i - 1, j, :] - rc)

                u13 = func.linear_interpolation(d1, d3, u[i - 1, j - 1], u[i, j])
                u24 = func.linear_interpolation(d2, d4, u[i, j - 1], u[i - 1, j])
                u_cell = (u13 + u24) * 0.5

                v13 = func.linear_interpolation(d1, d3, v[i - 1, j - 1], v[i, j])
                v24 = func.linear_interpolation(d2, d4, v[i, j - 1], v[i - 1, j])
                v_cell = (v13 + v24) * 0.5

                p13 = func.linear_interpolation(d1, d3, p[i - 1, j - 1], p[i, j])
                p24 = func.linear_interpolation(d2, d4, p[i, j - 1], p[i - 1, j])
                p_cell = (p13 + p24) * 0.5

                self.__V[i, j, 0] = u_cell
                self.__V[i, j, 1] = v_cell
                self.__p[i, j] = p_cell

        for i in range(1, ni):
            d1 = LA.norm(self.__nodes[i - 1, 0, :] - rc)
            d2 = LA.norm(self.__nodes[i, 0, :] - rc)
            u_cell = func.linear_interpolation(d1, d2, u[i - 1, 0], u[i, 0])
            v_cell = func.linear_interpolation(d1, d2, v[i - 1, 0], v[i, 0])
            p_cell = func.linear_interpolation(d1, d2, p[i - 1, 0], p[i, 0])
            self.__V[i, 0, 0] = u_cell
            self.__V[i, 0, 1] = v_cell
            self.__p[i, 0] = p_cell

            d1 = LA.norm(self.__nodes[i - 1, -1, :] - rc)
            d2 = LA.norm(self.__nodes[i, -1, :] - rc)
            u_cell = func.linear_interpolation(d1, d2, u[i - 1, -1], u[i, -1])
            v_cell = func.linear_interpolation(d1, d2, v[i - 1, -1], v[i, -1])
            p_cell = func.linear_interpolation(d1, d2, p[i - 1, -1], p[i, -1])
            self.__V[i, -1, 0] = u_cell
            self.__V[i, -1, 1] = v_cell
            self.__p[i, -1] = p_cell

        for j in range(1, nj):
            d1 = LA.norm(self.__nodes[0, j - 1, :] - rc)
            d2 = LA.norm(self.__nodes[0, j, :] - rc)
            u_cell = func.linear_interpolation(d1, d2, u[0, j - 1], u[0, j])
            v_cell = func.linear_interpolation(d1, d2, v[0, j - 1], v[0, j])
            p_cell = func.linear_interpolation(d1, d2, p[0, j - 1], p[0, j])
            self.__V[0, j, 0] = u_cell
            self.__V[0, j, 1] = v_cell
            self.__p[0, j] = p_cell

            d1 = LA.norm(self.__nodes[-1, j - 1, :] - rc)
            d2 = LA.norm(self.__nodes[-1, j, :] - rc)
            u_cell = func.linear_interpolation(d1, d2, u[-1, j - 1], u[-1, j])
            v_cell = func.linear_interpolation(d1, d2, v[-1, j - 1], v[-1, j])
            p_cell = func.linear_interpolation(d1, d2, p[-1, j - 1], p[-1, j])
            self.__V[-1, j, 0] = u_cell
            self.__V[-1, j, 1] = v_cell
            self.__p[-1, j] = p_cell

        self.__init_field_operator()

    def is_boundary(self, i, j):
        """
        Method is_boundary accepts i,j indicies of cell and return true if cell is a boundary one
        and false else

        :param i: i-index of cell
        :param j: j-index of cell
        :return: cell boundary status
        """
        if i == 0 or j == 0 or i == self.__ni or j == self.__nj:
            return True
        return False

    def calculate_grad(self, mode, iter=1):
        if mode == 0:
            grad_calc = self.__green_gauss_scalar
            grad_calc_vect = self.__green_gauss_vector
        if mode == 1:
            grad_calc = self.__least_squares_scalar
            grad_calc_vect = self.__green_gauss_vector
            iter = 1

        ni = self.__ni
        nj = self.__nj
        self.__grad_p[:, :, :] = 0.0
        self.__grad_T[:, :, :] = 0.0
        self.__grad_u[:, :, :] = 0.0
        self.__grad_v[:, :, :] = 0.0

        for gg_iter in range(iter):
            for i in range(1, ni):
                for j in range(1, nj):
                    grad_calc(i, j, self.__p, self.__grad_p)
                    grad_calc(i, j, self.__T, self.__grad_T)
                    grad_calc_vect(i, j, self.__V, self.__grad_u, self.__grad_v)

    def calculate_convective_div(self, mode):
        ni = self.__ni
        nj = self.__nj
        self.__div_TV[:, :] = 0.0
        self.__div_pV[:, :] = 0.0
        self.__div_Vu[:, :] = 0.0
        self.__div_Vv[:, :] = 0.0
        for i in range(1, ni):
            for j in range(1, nj):
                self.__conv_div_scalar(
                    i, j, self.__p, self.__grad_p, self.__div_pV, mode=mode
                )
                self.__conv_div_scalar(
                    i, j, self.__T, self.__grad_T, self.__div_TV, mode=mode
                )
                self.__conv_div_vector(
                    i,
                    j,
                    self.__V,
                    self.__grad_u,
                    self.__grad_v,
                    self.__div_Vu,
                    self.__div_Vv,
                    mode=mode,
                )

    def calculate_div(self):
        ni = self.__ni
        nj = self.__nj
        self.__div_V[:, :] = 0.0
        for i in range(1, ni):
            for j in range(1, nj):
                self.__divergence(
                    i, j, self.__V, self.__grad_u, self.__grad_v, self.__div_V
                )

    def calculate_curl(self):
        ni = self.__ni
        nj = self.__nj
        self.__curl_Vz[:, :] = 0.0
        for i in range(1, ni):
            for j in range(1, nj):
                self.__curl(
                    i, j, self.__V, self.__grad_u, self.__grad_v, self.__curl_Vz
                )

    def calculate_laplacian(self):
        ni = self.__ni
        nj = self.__nj
        self.__laplacian_p[:, :] = 0.0
        self.__laplacian_T[:, :] = 0.0
        self.__laplacian_u[:, :] = 0.0
        self.__laplacian_v[:, :] = 0.0
        for i in range(1, ni):
            for j in range(1, nj):
                self.__laplacian_scalar(
                    i, j, self.__p, self.__grad_p, self.__laplacian_p
                )
                self.__laplacian_scalar(
                    i, j, self.__T, self.__grad_T, self.__laplacian_T
                )
                self.__laplacian_vector(
                    i,
                    j,
                    self.__V,
                    self.__grad_u,
                    self.__grad_v,
                    self.__laplacian_u,
                    self.__laplacian_v,
                )

    def explicit_solve(self):
        CFL = 0.5
        min_vol = np.min(self.__cell_volume[:, :])
        v_ref = 1.0
        dt = np.sqrt(min_vol) * CFL / v_ref
        Re = 1000.0
        Pr = 1.0
        iter = 500

        for j in range(1, self.__nj):
            self.__T[0, j] = self.__T[1, j]
            self.__T[-1, j] = self.__T[-2, j]
        for i in range(1, self.__ni):
            self.__T[i, 0] = -self.__T[i, 1]
            self.__T[i, -1] = 0.2 - self.__T[i, -2]

        T = np.zeros((self.__ni + 1, self.__nj + 1), dtype=float)
        self.calculate_grad(mode=1)
        self.calculate_div()
        self.calculate_convective_div(mode=2)
        self.calculate_curl()
        self.calculate_laplacian()

        for n in range(iter):
            max_res_T = None

            self.__div_TV[:, :] = 0.0
            self.__laplacian_T[:, :] = 0.0
            for i in range(1, self.__ni):
                for j in range(1, self.__nj):
                    self.__conv_div_scalar(
                        i, j, self.__T, self.__grad_T, self.__div_TV, mode=2
                    )
                    self.__laplacian_scalar(
                        i, j, self.__T, self.__grad_T, self.__laplacian_T
                    )
                    res_T = self.__laplacian_T[i, j] / (Re * Pr) - self.__div_TV[i, j]
                    if max_res_T is None or np.abs(res_T) > max_res_T:
                        max_res_T = np.abs(res_T)
                        vol = self.__cell_volume[i - 1, j - 1]
                    T[i, j] = self.__T[i, j] + res_T * dt

            self.__T[:, :] = T[:, :]

            for j in range(1, self.__nj):
                self.__T[0, j] = self.__T[1, j]
                self.__T[-1, j] = self.__T[-2, j]
            for i in range(1, self.__ni):
                self.__T[i, 0] = -self.__T[i, 1]
                self.__T[i, -1] = 0.2 - self.__T[i, -2]

            self.__grad_T[:, :, :] = 0.0
            for i in range(1, self.__ni):
                for j in range(1, self.__nj):
                    self.__least_squares_scalar(i, j, self.__T, self.__grad_T)

            print(f"{n} {max_res_T * vol:.11e}")
