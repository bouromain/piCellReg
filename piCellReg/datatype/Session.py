import os.path as op
from dataclasses import dataclass

import bottleneck as bn
import numpy as np
from piCellReg.io.load import find_file_rec
from piCellReg.registration.utils import shift_coord
from scipy import sparse


class Base(object):
    def __post_init__(self):
        # see https://stackoverflow.com/a/59987363
        # just intercept the __post_init__ calls so they
        # aren't relayed to `object`
        pass


"""
dict_keys(['ypix', 'lam', 'xpix', 'mrs', 'mrs0', 'compact', 'med', 'npix', 'footprint', 'npix_norm', 'overlap', 'ipix', 'radius', 'aspect_ratio', 'skew', 'std'])
"""


class Roi:
    def __init__(
        self,
        x_pix=None,
        y_pix=None,
        lam=None,
        x_center=None,
        y_center=None,
        Lx=None,
        Ly=None,
    ):
        self._x_pix = x_pix
        self._y_pix = y_pix
        self._lam = lam
        self._x_center = x_center
        self._y_center = y_center


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
    _maxproj: np.ndarray = None
    _Vcor: np.ndarray = None
    _Lx: int = None
    _Ly: int = None

    # variable to set or calculated
    _x_offset: float = None
    _y_offset: float = None
    _rotation: float = None

    def __post_init__(self):
        super().__post_init__()
        # if the object is initialized with a correct stat path
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
            self._maxproj = ops["max_proj"]
            self._Vcor = ops["Vcorr"]
            self._diameter = ops["diameter"]
            self._Lx = ops["Lx"]
            self._Ly = ops["Ly"]

            # extract iscell
            if op.exists(self._iscell_path):
                self._iscell = np.load(self._iscell_path, allow_pickle=True)[:, 0]
            else:
                self._iscell = np.ones((self.n_cells))

    def __repr__(self) -> str:
        return f"{type(self).__name__} object with {self.n_cells} ROI"

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
    def to_hot_mat(self, x_shift: float = 0, y_shift: float = 0, theta=0):
        # return logical
        out = np.zeros((self.n_cells, self.Ly, self.Lx), dtype=bool)
        idx_cell = [np.ones_like(tmp) * it for it, tmp in enumerate(self._x_pix)]
        idx_cell = np.concatenate(idx_cell)
        x_pix = np.concatenate(self._x_pix)
        y_pix = np.concatenate(self._y_pix)

        if (x_shift != 0 and y_shift != 0) | theta != 0:
            origin = (self._Lx / 2, self._Ly / 2)
            x_pix, y_pix = shift_coord(x_pix, y_pix, x_shift, y_shift, origin, theta)

        # we could do something a bit more sophisticated here
        x_pix = np.floor(x_pix).astype(np.int32)
        y_pix = np.floor(y_pix).astype(np.int32)
        out[idx_cell, y_pix, x_pix] = True
        return out

    def to_sparse_hot_mat(self, x_shift: float = 0, y_shift: float = 0, theta=0):
        # return logical
        idx_cell = [np.ones_like(tmp) * it for it, tmp in enumerate(self._x_pix)]
        idx_cell = np.concatenate(idx_cell)
        x_pix = np.concatenate(self._x_pix)
        y_pix = np.concatenate(self._y_pix)

        ##
        if (x_shift != 0 and y_shift != 0) | theta != 0:
            origin = (self._Lx / 2, self._Ly / 2)
            x_pix, y_pix = shift_coord(x_pix, y_pix, x_shift, y_shift, origin, theta)

        # we could do something a bit more sophisticated here
        x_pix = np.floor(x_pix).astype(np.int32)
        y_pix = np.floor(y_pix).astype(np.int32)

        ###
        # first linearize totally the index to 1d
        # this step can be done more simply I guess by saying that:
        #  (z,y,x) = (z, i) with i = y * size_in_y + x
        #  But I need to check if I should need to switch x and y depending on F or C order
        lin_idx = np.ravel_multi_index(
            np.vstack((idx_cell, y_pix, x_pix)), (self.n_cells, self.Ly, self.Lx)
        )
        # then reshape it to 2D (n_cells, numel_image)
        idx = np.unravel_index(lin_idx, (self.n_cells, self.Ly * self.Lx))

        return sparse.csr_matrix(
            (np.ones_like(idx[0]), (idx[0], idx[1])),
            shape=(self.n_cells, self.Ly * self.Lx),
            dtype=bool,
        )

    def to_lam_mat(self, x_shift: float = 0, y_shift: float = 0, theta=0):
        # return fluorescence intensity
        out = np.zeros((self.n_cells, self.Ly, self.Lx), dtype=np.float32)
        idx_cell = [np.ones_like(tmp) * it for it, tmp in enumerate(self._x_pix)]
        idx_cell = np.concatenate(idx_cell)
        x_pix = np.concatenate(self._x_pix)
        y_pix = np.concatenate(self._y_pix)

        # we need to shift and interpolate data
        # we could do something a bit more sophisticated here
        if (x_shift != 0 and y_shift != 0) | theta != 0:
            origin = (self._Lx / 2, self._Ly / 2)
            x_pix, y_pix = shift_coord(x_pix, y_pix, x_shift, y_shift, origin, theta)

        x_pix = np.floor(x_pix).astype(np.int32)
        y_pix = np.floor(y_pix).astype(np.int32)
        ##
        out[idx_cell, y_pix, x_pix] = np.concatenate(self._lam)
        return out

    def to_sparse_lam_mat(self, x_shift: float = 0, y_shift: float = 0, theta=0):
        # return fluorescence intensity
        idx_cell = [np.ones_like(tmp) * it for it, tmp in enumerate(self._x_pix)]
        idx_cell = np.concatenate(idx_cell)
        x_pix = np.concatenate(self._x_pix)
        y_pix = np.concatenate(self._y_pix)
        data = np.concatenate(self._lam)

        ##
        if (x_shift != 0 and y_shift != 0) | theta != 0:
            origin = (self._Lx / 2, self._Ly / 2)
            x_pix, y_pix = shift_coord(x_pix, y_pix, x_shift, y_shift, origin, theta)

        # we could do something a bit more sophisticated here
        x_pix = np.floor(x_pix).astype(np.int32)
        y_pix = np.floor(y_pix).astype(np.int32)

        ###
        # first linearized totally the index to 1d
        # this step can be done more simply I guess by saying that:
        #  (z,y,x) = (z, i) with i = y * size_in_y + x
        #  But I need to check if I should need to switch x and y depending on F or C order
        lin_idx = np.ravel_multi_index(
            np.vstack((idx_cell, y_pix, x_pix)), (self.n_cells, self.Ly, self.Lx)
        )
        # then reshape it to 2D (n_cells, numel_image)
        idx = np.unravel_index(lin_idx, (self.n_cells, self.Ly * self.Lx))

        return sparse.csr_matrix(
            (data, (idx[0], idx[1])),
            shape=(self.n_cells, self.Ly * self.Lx),
            dtype=data.dtype,
        )

    def get_roi(self, n=0, margin=10):
        if n > self.n_cells:
            raise IndexError(
                f"Requested value n = {n} greater than the number of cells {self.n_cells}"
            )

        (x_0, y_0, x_end, y_end) = _bounding_box(
            self._x_pix[n], self._y_pix[n], margin_x=margin
        )
        return self._mean_image_e[y_0:y_end, x_0:x_end]

    def get_projection(
        self, mask=None, x_shift: float = 0, y_shift: float = 0, theta=0
    ):
        if mask is None:
            mask = np.array(self._iscell, dtype=bool)

        return bn.nansum(
            self.to_hot_mat(x_shift=x_shift, y_shift=y_shift, theta=theta)[mask, :, :],
            axis=0,
        )


@dataclass
class SessionList:
    """
    Wrapper to group all sessions in a root path
    """

    # general infos
    _root_path: str = None
    _sessions = []

    # load all available sessions in a root path
    def load_from_s2p(self, fpath: str = None):
        self._root_path = fpath
        list_stat = find_file_rec(self._root_path, "stat.npy")
        self._sessions = [Session(p) for p in list_stat]

        return self

    @property
    def n_session(self):
        return len(self._sessions)

    def __len__(self):
        return self.n_session

    def __repr__(self) -> str:
        return f"{type(self).__name__} object with {self.n_session} sessions"

    @property
    def isempty(self):
        if self._sessions == [] or self._sessions is None:
            return True
        else:
            return False

    def __getitem__(self, idx):
        if self.isempty:
            return self

        if isinstance(idx, (int, slice, np.int32, np.int64, np.ndarray)):
            return self._sessions[idx]
        else:
            raise TypeError(f"Unsupported indexing type {type(idx)}")

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        index = self._index
        if index > self.n_session - 1:
            raise StopIteration

        self._index += 1
        return self._sessions[index]


def _bounding_box(
    x: np.ndarray, y: np.ndarray, margin_x: int = 5, margin_y: int = None
):
    """
         _bounding_box return bounding box  of provided coordinates 
        with a given margin

        Parameters
        ----------
        x : np.ndarray
            x coordinates
        y : np.ndarray
            x coordinates
        margin_x : int, optional
            margin we want in the x axis, by default 5
        margin_y : int, optional
            [description], by default the same than margin_x

        Returns
        -------
        tuple :
            bounding box coordinates, (x_0,y_0, x_end,y_end)
            [description]
        """

    if margin_y is None:
        margin_y = margin_x

    return (
        bn.nanmin(x) - margin_x,
        bn.nanmin(y) - margin_y,
        bn.nanmax(x) + margin_x,
        bn.nanmax(y) + margin_y,
    )

