import os

import numpy as np

from meshtools import metric
from meshtools import output as out
from meshtools import taskinit as inp
from spatial import divergence, gradient
from spatial import laplacian as lapl
from spatial import rotor


def p_init(
    x: float,
    y: float,
) -> float:
    return x * x + y * y


def gradp_exact_init_x(x: float, y: float) -> float:
    return 1.0


def gradp_exact_init_y(
    x: float,
    y: float,
) -> float:
    return 1.0


def v_init_x(
    x: float,
    y: float,
) -> float:
    # return x
    return (1 + x) * (1 + y)


def v_init_y(
    x: float,
    y: float,
) -> float:
    # return y
    return x * y


def div_exact_init(x: float, y: float) -> float:
    # mode = 0
    # return 1 + x + y
    return x * x + 4 * x * y + 2 * x + y * y + 2 * y + 1


def rot_exact_init(
    x: float,
    y: float,
) -> float:
    # return 0.0
    return 1 + x - y


def laplacian_exact_init(
    x: float,
    y: float,
) -> float:
    return 4.0


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
        p[:, :] = p_init(cell_center[:, :, 0], cell_center[:, :, 1])
        gradp_exact = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        gradp_error = np.zeros((ni + 1, nj + 1, 2), dtype=float)
        gradp_exact[:, :, 0] = gradp_exact_init_x(
            cell_center[:, :, 0], cell_center[:, :, 1]
        )
        gradp_exact[:, :, 1] = gradp_exact_init_y(
            cell_center[:, :, 0], cell_center[:, :, 1]
        )
        v[:, :, 0] = v_init_x(cell_center[:, :, 0], cell_center[:, :, 1])
        v[:, :, 1] = v_init_y(cell_center[:, :, 0], cell_center[:, :, 1])
        div_exact = np.zeros((ni + 1, nj + 1), dtype=float)
        div_exact[:, :] = div_exact_init(cell_center[:, :, 0], cell_center[:, :, 1])
        div_error = np.zeros((ni + 1, nj + 1), dtype=float)
        rot_exact = np.zeros((ni + 1, nj + 1), dtype=float)
        rot_exact[:, :] = rot_exact_init(cell_center[:, :, 0], cell_center[:, :, 1])
        rot_error = np.zeros((ni + 1, nj + 1), dtype=float)
        laplacian_exact = np.zeros((ni + 1, nj + 1), dtype=float)
        laplacian_exact[:] = laplacian_exact_init(
            cell_center[:, :, 0], cell_center[:, :, 1]
        )
        laplacian_error = np.zeros((ni + 1, nj + 1), dtype=float)
    else:
        with open(taskinit.data, "r") as in_data:
            in_data.readline()
            in_data.readline()
            for j in range(nj + 1):
                for i in range(ni + 1):
                    line = in_data.readline().strip().split()
                    _, _, v[i, j, 0], v[i, j, 1], _, p[i, j], *_ = map(float, line)

    print("\nGradient calculation:")
    for i in range(1, taskinit.gauss_iter + 1):
        gradient.calc_gradient(
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
            gradp_error[1:ni, 1:nj, :] = np.absolute(
                (gradp_exact[1:ni, 1:nj, :] - gradp[1:ni, 1:nj, :])
                / gradp_exact[1:ni, 1:nj, :]
            )
            print(f"i = {i}:")
            print("Maximum X-Gradient error: ", np.amax(gradp_error[:, :, 0]))
            print("Maximum Y-Gradient error: ", np.amax(gradp_error[:, :, 1]))

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
        div_error[1:ni, 1:nj] = np.absolute(
            (div_exact[1:ni, 1:nj] - div[1:ni, 1:nj]) / div[1:ni, 1:nj]
        )
        print("Maximum divergence error: ", np.amax(div_error[:, :]))

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
        rot_error[1:ni, 1:nj] = np.absolute(
            (rot_exact[1:ni, 1:nj] - rot[1:ni, 1:nj]) / rot_exact[1:ni, 1:nj]
        )
        print("Maximum rotor error: ", np.amax(rot_error[:, :]))

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
        laplacian_error[1:ni, 1:nj] = np.absolute(
            (laplacian_exact[1:ni, 1:nj] - laplacian[1:ni, 1:nj])
            / laplacian_exact[1:ni, 1:nj]
        )
        print("Maximum laplacian error: ", np.amax(laplacian_error[:, :]))

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
