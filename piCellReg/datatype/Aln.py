from matplotlib.pyplot import axis
import numpy as np
from dataclasses import dataclass
from piCellReg.io.load import find_file_rec
from piCellReg.datatype.Session import Session, Base
from skimage.registration import phase_cross_correlation
from itertools import combinations
import tqdm
import bottleneck as bn


class Registration(Base):
    def __init__(self, n_session):
        self._method = None

        self._error_mat = np.empty((n_session, n_session))
        self._error_mat.fill(np.nan)

        self._x_shifts = np.empty((n_session, n_session))
        self._x_shifts.fill(np.nan)

        self._y_shifts = np.empty((n_session, n_session))
        self._y_shifts.fill(np.nan)

        self._rot_mat = np.empty((n_session, n_session))
        self._rot_mat.fill(np.nan)

    def isempty(self):
        if self._x_shifts == [] or self._x_shifts is None:
            return True
        else:
            return False

    def best_ref(self):
        """
        return index of the best reference session
        The best reference session is the one that will minimize the
        registration error with all the other pairs
        """
        if np.isnan(self._error_mat).all():
            return np.nan
        else:
            return bn.nanargmin(bn.nansum(self._error_mat, axis=0))


@dataclass
class Aln:
    """
    Wrapper to group all sessions in a root path
    """

    # general infos
    _root_path: str = None
    _sessions = []

    # registration
    _reference_session = None
    _registration = None

    # load all available sessions in a root path
    def __post_init__(self):
        list_stat = find_file_rec(self._root_path, "stat.npy")
        print(f"{len(list_stat)} sessions found, initialing ...")
        self._sessions = [Session(p) for p in list_stat]
        print(f"Alignment structure ready")

        # initialise the registration object properly
        self._registration = Registration(self.n_session)

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

    @property
    def ref_session(self):
        return self._reference_session

    @ref_session.setter
    def ref_session(self, value):

        if value > self.n_session - 1:
            raise ValueError(f" Value {value} greater than the number of sessions ")
        else:
            # here we should make a better check of the value type
            self._reference_session = int(value)

        return self

    def register(self):

        for i, j in tqdm.tqdm(combinations(np.arange(self.n_session), 2)):
            tmp = phase_cross_correlation(
                self[i]._mean_image_e, self[j]._mean_image_e, upsample_factor=100,
            )
            self._registration._y_shifts[i, j] = tmp[0][0]
            self._registration._x_shifts[i, j] = tmp[0][1]
            self._registration._error_mat[i, j] = tmp[1]

        # ensure matrices are symmetric
        up_tri_idx = np.triu_indices_from(self._registration._error_mat, 1)
        low_tri_idx = np.tril_indices_from(self._registration._error_mat, -1)

        self._registration._y_shifts[low_tri_idx] = self._registration._y_shifts[
            up_tri_idx
        ]
        self._registration._x_shifts[low_tri_idx] = self._registration._x_shifts[
            up_tri_idx
        ]
        self._registration._error_mat[low_tri_idx] = self._registration._error_mat[
            up_tri_idx
        ]

        return self

