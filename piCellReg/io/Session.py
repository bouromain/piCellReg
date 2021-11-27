from os import get_terminal_size
import os.path as op
from dataclasses import dataclass
from typing import ValuesView
import numpy as np

from registration.register import register_im


class Base(object):
    def __post_init__(self):
        # see https://stackoverflow.com/a/59987363
        # just intercept the __post_init__ calls so they
        # aren't relayed to `object`
        pass


@dataclass
class Session(Base):
    _stat_path: str = None
    _ops_path: str = None

    # basic infos on the session
    _cell_coord: np.ndarray = None
    _cell_lam: np.ndarray = None
    _is_cell: np.ndarray = None

    _mean_image: np.ndarray = None
    _mean_image_e: np.ndarray = None

    # variable to set or calculated
    _pixel_micro: float = None
    _offsets: tuple = None
    _rotation: float = None

    def __post_init__(self):
        super().__post_init__()
        # if the object is initialised with a correct stat path
        if self._stat_path is not None:
            # find ops path if needed
            if self._ops_path is None and op.exists(self._stat_path):
                self._ops_path = self._stat_path.replace("stat.npy", "ops.npy")

            # extract data from ops and stat
            stat = np.load(self._stat_path, allow_pickle=True)
            ops = np.load(self._ops_path, allow_pickle=True).item()

    @property
    def pixel_micro(self):
        return self._pixel_micro

    @pixel_micro.setter
    def pixel_micro(self, val):
        self._pixel_micro = val

    def register(self, ref_image):
        self._offsets, self._rotation = register_im(ref_image, self._mean_image)

    @property
    def cell_coord(self):
        return self._cell_coord

    @property
    def cell_lam(self):
        return self._cell_lam

