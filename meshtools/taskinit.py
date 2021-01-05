import os
from typing import Tuple


def is_integer(str_to_check: str) -> bool:
    try:
        int(str_to_check)
    except ValueError:
        return False
    else:
        return True


def is_float(str_to_check: str) -> bool:
    try:
        float(str_to_check)
    except ValueError:
        return False
    else:
        return True


class InputManager:
    """
    Class InputManager is designed to parse init_file data
    to program
    """

    __slots__ = "__storage"

    def __init__(self, file_path: str):
        self.__storage = {}
        with open(file_path, "r") as in_file:
            for line in in_file:
                line = line.strip().split("#", 1)[0]
                try:
                    key, val, *_ = line.strip().split("=", 1)
                except ValueError:
                    continue

                if not key.isidentifier():
                    raise ValueError(
                        f"Expected: Unacceptable attribute name({key}) is given!"
                    )
                if is_integer(val):
                    val = int(val)
                elif is_float(val):
                    val = float(val)
                self.__storage[key] = val
        self.__check_init()

    def __check_init(self):
        if not "mesh" in self.__storage:
            raise AttributeError("Error: Mesh file name not found!")
        if not isinstance(self.__storage["mesh"], str):
            raise TypeError("Error: Mesh file name expected to be string!")

        if not isinstance(self.__storage.get("grad", 0), int):
            raise TypeError(
                "Error: Gradient calculation approach id" + "expected to be integer!"
            )
        if not (0 <= self.__storage.get("grad", 0) <= 1):
            raise ValueError("Error: Unknown gradient calculation approach id!")

        if not isinstance(self.__storage.get("div_mode", 0), int):
            raise TypeError("Error: Divergence calc mode expected to be integer!")
        if not (0 <= self.__storage.get("div_mode", 0) <= 3):
            raise ValueError("Error: Unknown divergence calc mode!")

        if not isinstance(self.__storage.get("gauss_iter", 1), int):
            raise TypeError(
                "Error: Number of green gauss method's"
                + "iteration expected to be integer!"
            )
        if not self.__storage.get("gauss_iter", 1) > 0:
            raise ValueError(
                "Error: Number of green gauss method's"
                + "iteration expected to be positive!"
            )

        if not isinstance(self.__storage.get("data", ""), str):
            raise TypeError("Error: Data file name expected to be string!")

        if not isinstance(self.__storage.get("testfunc", 1), int):
            raise TypeError("Error: Test func id expected to be int!")
        if not (1 <= self.__storage.get("testfunc", 1) <= 3):
            raise ValueError("Error: Unknown test function id!")

    @property
    def msh(self) -> str:
        return self.__storage["mesh"]

    @property
    def div_mode(self) -> Tuple[int, str]:
        mode = self.__storage.get("div_mode", 0)
        if mode == 0:
            return mode, "standard operator"
        if mode == 1:
            return mode, "central scheme"
        if mode == 2:
            return mode, "first order upwind scheme"
        if mode == 3:
            return mode, "second order upwind scheme"

    @property
    def grad(self) -> Tuple[int, str]:
        grad_scheme = self.__storage.get("grad", 0)
        if grad_scheme == 0:
            return grad_scheme, "Green Gauss"
        if grad_scheme == 1:
            return grad_scheme, "Least Squares"

    @property
    def gauss_iter(self) -> int:
        return self.__storage.get("gauss_iter", 1)

    @property
    def data(self) -> str:
        return self.__storage.get("data", "")

    @property
    def outfile(self) -> str:
        return self.__storage.get("outfile", "data.plt")

    @property
    def testfunc(self) -> int:
        return self.__storage.get("testfunc", 1)
