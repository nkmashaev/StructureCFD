import os

import numpy as np

from meshtools import metric
from meshtools import output as out
from meshtools import taskinit as inp
from spatial import divergence, gradient
from spatial import laplacian as lapl
from spatial import rotor


def p_init(x: float, y: float, func_id: int = 1) -> float:
    if func_id == 1:
        return x + y
    if func_id == 2:
        return (x + y) ** 2
    if func_id == 3:
        return (x + y) ** 3


def gradp_exact_init_x(x: float, y: float, func_id: int = 1) -> float:
    if func_id == 1:
        return 1.0
    if func_id == 2:
        return 2.0 * (x + y)
    if func_id == 3:
        return 3.0 * (x + y) ** 2


def gradp_exact_init_y(x: float, y: float, func_id: int = 1) -> float:
    if func_id == 1:
        return 1.0
    if func_id == 2:
        return 2.0 * (x + y)
    if func_id == 3:
        return 3.0 * (x + y) ** 2


def v_init_x(x: float, y: float, func_id: int = 1) -> float:
    if func_id == 1:
        return x + 2 * y
    if func_id == 2:
        return (x - y) ** 2 + 3.0 * x * y
    if func_id == 3:
        return (x - y) ** 3 + 3.0 * x ** 2 * y


def v_init_y(x: float, y: float, func_id: int = 1) -> float:
    if func_id == 1:
        return 0.5 * x + y
    if func_id == 2:
        return (x + y) ** 2 + 3.0 * x * y
    if func_id == 3:
        return (x + y) ** 3 + 3.0 * x * y ** 2


def div_exact_init(x: float, y: float, func_id: int = 1, mode: int = 0) -> float:
    if func_id == 1:
        if mode == 0:
            return 2.0
        return 3.5 * x + 5.0 * y
    if func_id == 2:
        if mode == 0:
            return 7.0 * x + 3.0 * y
        return 11.0 * x ** 3 + 33.0 * x ** 2 * y + 29 * x * y ** 2 + 7.0 * y ** 3
    if func_id == 3:
        if mode == 0:
            return 6.0 * (x + y) ** 2
        return (
            3.0
            * (x + y) ** 2
            * (4.0 * x ** 3 + 9.0 * x ** 2 * y + 15.0 * x * y ** 2 + 2.0 * y ** 3)
        )


def rot_exact_init(x: float, y: float, func_id: int = 1) -> float:
    if func_id == 1:
        return -1.5
    if func_id == 2:
        return x + 3.0 * y
    if func_id == 3:
        return 3.0 * (x ** 2 + 3.0 * y ** 2)


def laplacian_exact_init(x: float, y: float, func_id: int = 1) -> float:
    if func_id == 1:
        return 0.0
    if func_id == 2:
        return 4.0
    if func_id == 3:
        return 12.0 * (x + y)


if __name__ == "__main__":
    # read initialization parameters
    input_name = os.path.join(os.path.dirname(__file__), "input.txt")
    print(f'Read initialization parameters from file: "{input_name}"')
    taskinit = inp.InputManager(input_name)
    print(f'Mesh file name is "{taskinit.msh}"')
    if not taskinit.data:
        to_init = True
    else:
        to_init = False
        print(f'Data file name is "{taskinit.data}"')

    print(f"Gradient calculation approach: {taskinit.grad[1]}")
    if taskinit.grad[0] == 0:
        print(f"Number of green gauss iterations equals {taskinit.gauss_iter}")
    print(f"Divergence calc method: {taskinit.div_mode[1]}")

    # read mesh file
    with open(taskinit.msh, "r") as in_file:
        size_list = in_file.readline().strip().split()
        ni = int(size_list[0])
        nj = int(size_list[1])
        x = np.zeros((ni, nj), dtype=float)
        y = np.zeros((ni, nj), dtype=float)
        cell_center = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        cell_volume = np.zeros((ni + 1, nj + 1), dtype=float)
        i_face_vector = np.zeros((ni, nj - 1, 2), dtype=float)
        i_face_center = np.zeros((ni, nj - 1, 2), dtype=float)
        j_face_vector = np.zeros((ni - 1, nj, 2), dtype=float)
        j_face_center = np.zeros((ni - 1, nj, 2), dtype=float)
        for j in range(nj):
            for i in range(ni):
                coord_list = in_file.readline().strip().split()
                x[i, j] = float(coord_list[0])
                y[i, j] = float(coord_list[1])

    metric.calc_metric(
        ni,
        nj,
        x,
        y,
        cell_center,
        cell_volume,
        i_face_center,
        i_face_vector,
        j_face_center,
        j_face_vector,
    )

    outp = out.OutputManager(x, y)

    p = np.zeros((ni + 1, nj + 1), dtype=float)
    gradp = np.zeros((ni + 1, nj + 1, 2), dtype=float)
    v = np.zeros((ni + 1, nj + 1, 2), dtype=float)
    div = np.zeros((ni + 1, nj + 1), dtype=float)
    rot = np.zeros((ni + 1, nj + 1), dtype=float)
    laplacian = np.zeros((ni + 1, nj + 1), dtype=float)

    if to_init:
        p[:, :] = p_init(cell_center[:, :, 0], cell_center[:, :, 1], taskinit.testfunc)
        gradp_exact = np.zeros((ni + 1, nj + 1, 2), dtype=float)

        gradp_exact[:, :, 0] = gradp_exact_init_x(
            cell_center[:, :, 0], cell_center[:, :, 1], taskinit.testfunc
        )
        gradp_exact[:, :, 1] = gradp_exact_init_y(
            cell_center[:, :, 0], cell_center[:, :, 1], taskinit.testfunc
        )
        v[:, :, 0] = v_init_x(
            cell_center[:, :, 0], cell_center[:, :, 1], taskinit.testfunc
        )
        v[:, :, 1] = v_init_y(
            cell_center[:, :, 0], cell_center[:, :, 1], taskinit.testfunc
        )
        div_exact = np.zeros((ni + 1, nj + 1), dtype=float)
        div_exact[:, :] = div_exact_init(
            cell_center[:, :, 0],
            cell_center[:, :, 1],
            taskinit.testfunc,
            taskinit.div_mode[0],
        )
        rot_exact = np.zeros((ni + 1, nj + 1), dtype=float)
        rot_exact[:, :] = rot_exact_init(
            cell_center[:, :, 0], cell_center[:, :, 1], taskinit.testfunc
        )
        laplacian_exact = np.zeros((ni + 1, nj + 1), dtype=float)
        laplacian_exact[:] = laplacian_exact_init(
            cell_center[:, :, 0], cell_center[:, :, 1], taskinit.testfunc
        )
        # laplacian_error = np.zeros((ni + 1, nj + 1), dtype=float)
    else:
        with open(taskinit.data, "r") as in_data:
            in_data.readline()
            in_data.readline()
            for j in range(nj + 1):
                for i in range(ni + 1):
                    line = in_data.readline().strip().split()
                    _, _, v[i, j, 0], v[i, j, 1], _, p[i, j], *_ = map(float, line)

    print("\nGradient calculation:")
    if taskinit.grad[0] == 0:
        calc_gradient = gradient.green_gauss
    elif taskinit.grad[0] == 1:
        calc_gradient = gradient.least_squares
    for i in range(1, taskinit.gauss_iter + 1):
        calc_gradient(
            ni,
            nj,
            p,
            gradp,
            cell_volume,
            cell_center,
            i_face_center,
            i_face_vector,
            j_face_center,
            j_face_vector,
        )
        if to_init:
            gradp_error = np.absolute(
                np.divide(
                    gradp_exact - gradp,
                    gradp_exact,
                    out=np.zeros_like(gradp_exact - gradp),
                    where=np.absolute(gradp_exact) > 1.0e-14,
                )
            )
            if taskinit.grad[0] == 0:
                print(f"i = {i}:")
            print("Maximum X-Gradient error: ", np.amax(gradp_error[1:ni, 1:nj, 0]))
            print("Maximum Y-Gradient error: ", np.amax(gradp_error[1:ni, 1:nj, 1]))

    print("\nCalculate divergence:")
    divergence.calc_divergence(
        ni,
        nj,
        v,
        div,
        p,
        gradp,
        cell_volume,
        cell_center,
        i_face_center,
        i_face_vector,
        j_face_center,
        j_face_vector,
        taskinit.div_mode[0],
    )
    if to_init:
        div_error = np.absolute(
            np.divide(
                div_exact - div,
                div_exact,
                out=np.zeros_like(div_exact - div),
                where=np.absolute(div_exact) > 1.0e-14,
            )
        )
        print("Maximum divergence error: ", np.amax(div_error[1:ni, 1:nj]))

    print("\nCalculate rotor:")
    rot_error = np.zeros((ni + 1, nj + 1), dtype=float)
    rotor.calc_rotor(
        ni,
        nj,
        v,
        rot,
        cell_volume,
        cell_center,
        i_face_center,
        i_face_vector,
        j_face_center,
        j_face_vector,
    )
    if to_init:
        rot_error = np.absolute(
            np.divide(
                rot_exact - rot,
                rot_exact,
                out=np.zeros_like(rot_exact - rot),
                where=np.absolute(rot_exact) > 1.0e-14,
            )
        )
        print("Maximum rotor error: ", np.amax(rot_error[1:ni, 1:nj]))

    print("\nCalculate laplacian:")
    lapl.calc_laplacian(
        ni,
        nj,
        p,
        gradp,
        laplacian,
        cell_volume,
        cell_center,
        i_face_center,
        i_face_vector,
        j_face_center,
        j_face_vector,
    )
    if to_init:
        laplacian_error = np.absolute(
            np.divide(
                laplacian_exact - laplacian,
                laplacian_exact,
                out=np.zeros_like(laplacian_exact - laplacian),
                where=np.absolute(laplacian_exact) > 1.0e-14,
            )
        )
        print("Maximum laplacian error: ", np.amax(laplacian_error[1:ni, 1:nj]))

    print(f"Writing data in {taskinit.outfile}...")
    outp.save_scalar("Pressure", p)
    outp.save_scalar("X-Gradient Pressure", gradp[:, :, 0])
    outp.save_scalar("Y-Gradient Pressure", gradp[:, :, 1])
    outp.save_scalar("Laplacian Pressure", laplacian[:, :])
    outp.save_scalar("X-Velocity", v[:, :, 0])
    outp.save_scalar("Y-Velocity", v[:, :, 1])
    outp.save_scalar("Divergence", div[:, :])
    outp.save_scalar("Rotor-Z Velocity", rot[:, :])

    if to_init:
        outp.save_scalar("X-Gradient Error", gradp_error[:, :, 0])
        outp.save_scalar("Y-Gradient Error", gradp_error[:, :, 1])
        outp.save_scalar("Laplacian Error", laplacian_error[:, :])
        outp.save_scalar("Divergence Error", div_error[:, :])
        outp.save_scalar("Rotor-Z Error", rot_error[:, :])
    outp.output(taskinit.outfile)
    print("Done!")
