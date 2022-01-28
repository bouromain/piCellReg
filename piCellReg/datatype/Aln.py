from matplotlib.pyplot import axis
import numpy as np
from dataclasses import dataclass
from piCellReg.io.load import find_file_rec
from piCellReg.datatype.Session import Session, Base
from skimage.registration import phase_cross_correlation
from itertools import combinations
import tqdm
import bottleneck as bn


@dataclass
class Aln:
    """
    Wrapper to group all sessions in a root path
    """

    # general infos
    _root_path: str = None
    _sessions = []

    # load all available sessions in a root path
    def __post_init__(self):
        list_stat = find_file_rec(self._root_path, "stat.npy")
        print(f"{len(list_stat)} sessions found, initialing ...")
        self._sessions = [Session(p) for p in list_stat]
        print(f"Alignment structure ready")

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
