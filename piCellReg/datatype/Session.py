import os.path as op
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from piCellReg.registration.utils import shift_coord


class Base(object):
    def __post_init__(self):
        # see https://stackoverflow.com/a/59987363
        # just intercept the __post_init__ calls so they
        # aren't relayed to `object`
        pass


"""
dict_keys(['ypix', 'lam', 'xpix', 'mrs', 'mrs0', 'compact', 'med', 'npix', 'footprint', 'npix_norm', 'overlap', 'ipix', 'radius', 'aspect_ratio', 'skew', 'std'])
"""


@dataclass
class Session(Base):
    _stat_path: str = None
    _ops_path: str = None
    _iscell_path: str = None

    # basic infos on the session
    _idx: int = None  # index of the session
    _x_pix: list = None  # x pixels of the
    _y_pix: list = None
    _lam: list = None
    _x_center: np.ndarray = None
    _y_center: np.ndarray = None
    _is_cell: np.ndarray = None

    _mean_image: np.ndarray = None
    _mean_image_e: np.ndarray = None
    _Lx: int = None
    _Ly: int = None

    # variable to set or calculated
    _diameter: float = None
    _x_offset: float = None
    _y_offset: float = None
    _rotation: float = None

    def __post_init__(self):
        super().__post_init__()
        # if the object is initialised with a correct stat path
        if self._stat_path is not None:
            # find ops path if needed
            if self._ops_path is None and op.exists(self._stat_path):
                self._ops_path = self._stat_path.replace("stat.npy", "ops.npy")

            if self._iscell_path is None and op.exists(self._stat_path):
                self._iscell_path = self._stat_path.replace("stat.npy", "iscell.npy")

            # extract data from stat
            stat = np.load(self._stat_path, allow_pickle=True)
            self._x_pix = [s["xpix"][~s["overlap"]] for s in stat]
            self._y_pix = [s["ypix"][~s["overlap"]] for s in stat]
            self._lam = [s["lam"][~s["overlap"]] for s in stat]
            self._y_center = np.array([s["med"][0] for s in stat])
            self._x_center = np.array([s["med"][1] for s in stat])

            # extract data from ops
            ops = np.load(self._ops_path, allow_pickle=True).item()
            self._mean_image = ops["meanImg"]
            self._mean_image_e = ops["meanImgE"]
            self._diameter = ops["diameter"]
            self._Lx = ops["Lx"]
            self._Ly = ops["Ly"]

            # extract iscell
            self._iscell = np.load(self._iscell_path, allow_pickle=True)[:, 0]

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, val):
        self._diameter = val

    @property
    def n_cells(self):
        return len(self._x_pix)

    @property
    def Lx(self):
        return self._Lx

    @property
    def Ly(self):
        return self._Ly

    @property
    def x_pix_off(self):
        return self._x_pix - self._x_offset

    @property
    def y_pix_off(self):
        return self._y_pix - self._y_offset[1]

    # methods
    def to_hot_mat(self, shifts=np.array([0, 0]), theta=0):
        # return logical
        out = np.zeros((self.n_cells, self.Ly, self.Lx), dtype=bool)
        idx_cell = [np.ones_like(tmp) * it for it, tmp in enumerate(self._x_pix)]
        idx_cell = np.concatenate(idx_cell)
        x_pix = np.concatenate(self._x_pix)
        y_pix = np.concatenate(self._y_pix)

        if (shifts != 0).all() | theta != 0:
            origin = (self._Lx / 2, self._Ly / 2)
            x_pix, y_pix = shift_coord(
                x_pix, y_pix, shifts[0], shifts[1], origin, theta
            )

        # we could do something a bit more sophisticated here
        x_pix = np.round(x_pix).astype(np.int32)
        y_pix = np.round(y_pix).astype(np.int32)
        out[idx_cell, y_pix, x_pix] = True
        return out

    def to_lam_mat(self, shifts=np.array([0, 0]), theta=0):
        # return fluorescence intensity
        out = np.zeros((self.n_cells, self.Ly, self.Lx), dtype=bool)
        idx_cell = [np.ones_like(tmp) * it for it, tmp in enumerate(self._x_pix)]
        idx_cell = np.concatenate(idx_cell)
        x_pix = np.concatenate(self._x_pix)
        y_pix = np.concatenate(self._y_pix)

        if (shifts != 0).all() | theta != 0:
            origin = (self._Lx / 2, self._Ly / 2)
            x_pix, y_pix = shift_coord(
                x_pix, y_pix, shifts[0], shifts[1], origin, theta
            )

        # we could do something a bit more sophisticated here
        x_pix = np.round(x_pix).astype(np.int32)
        y_pix = np.round(y_pix).astype(np.int32)
        out[idx_cell, y_pix, x_pix] = np.concatenate(self._lam)
        return out
