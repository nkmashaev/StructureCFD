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
                key, val = line.strip().split("=", 1)

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
        if not "div_mode" in self.__storage:
            raise AttributeError("Error: Divergence calc mode not found!")
        if not isinstance(self.__storage["div_mode"], int):
            raise TypeError("Error: Divergence calc mode expected to be integer!")
        if self.__storage["div_mode"] < 0 or self.__storage["div_mode"] > 3:
            raise ValueError("Error: Unknown divergence calc mode!")
        if not "gauss_iter" in self.__storage:
            raise AttributeError(
                "Error: Number of green gauss method's iteration not found!"
            )
        if not isinstance(self.__storage["gauss_iter"], int):
            raise TypeError(
                "Error: Number of green gauss method's"
                + "iteration expected to be integer!"
            )
        if "data" in self.__storage and not isinstance(self.__storage["data"], str):
            raise TypeError("Error: Data file name expected to be string!")

    @property
    def msh(self) -> str:
        return self.__storage["mesh"]

    @property
    def div_mode(self) -> Tuple[int, str]:
        mode = self.__storage["div_mode"]
        if mode == 0:
            return mode, "standard operator"
        if mode == 1:
            return mode, "central scheme"
        if mode == 2:
            return mode, "first order upwind scheme"
        if mode == 3:
            return mode, "second order upwind scheme"

    @property
    def gauss_iter(self) -> int:
        return self.__storage["gauss_iter"]

    @property
    def data(self) -> str:
        if "data" in self.__storage:
            return self.__storage["data"]
        return ""

    @property
    def outfile(self) -> str:
        if "outfile" in self.__storage:
            return self.__storage["outfile"]
        return "data.plt"
