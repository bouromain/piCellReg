import os.path as op
from dataclasses import dataclass
import numpy as np
from registration.register import register_im


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

    # basic infos on the session
    _cell_x_pix: np.ndarray = None
    _cell_y_pix: np.ndarray = None
    _cell_lam: np.ndarray = None
    _is_cell: np.ndarray = None

    _mean_image: np.ndarray = None
    _mean_image_e: np.ndarray = None

    # variable to set or calculated
    _diameter: float = None
    _offsets: tuple = None
    _rotation: float = None

    def __post_init__(self):
        super().__post_init__()
        # if the object is initialised with a correct stat path
        if self._stat_path is not None:
            # find ops path if needed
            if self._ops_path is None and op.exists(self._stat_path):
                self._ops_path = self._stat_path.replace("stat.npy", "ops.npy")

            # extract data from stat
            stat = np.load(self._stat_path, allow_pickle=True)
            self._cell_x_pix = [s["xpix"] for s in stat]
            self._cell_y_pix = [s["ypix"] for s in stat]
            self._cell_lam = [s["lam"] for s in stat]

            # extract data from ops
            ops = np.load(self._ops_path, allow_pickle=True).item()
            self._mean_image = ops["meanImg"]
            self._mean_image_e = ops["meanImgE"]
            self._diameter = ops["diameter"]

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, val):
        self._diameter = val

    def register(self, ref_image):
        self._offsets, self._rotation = register_im(ref_image, self._mean_image)

