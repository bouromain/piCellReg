from dataclasses import dataclass
import itertools
from piCellReg.datatype.SessionPair import SessionPair
import numpy as np
from itertools import combinations
from piCellReg.utils.fit import fit_center_distribution, calc_psame, psame_matrix


@dataclass
class SessionPairList:
    """
    Wrapper to group SessionsPairs
    """

    # general infos
    _sessions_pairs = []

    # load all available sessions in a root path
    def from_list(self, session_pairs_list: list):
        self._sessions_pairs = session_pairs_list

    def from_SessionList(self, sess_list):

        self._sessions_pairs = [
            SessionPair(s0, s1, id_s0=i0, id_s1=i1)
            for ((i0, s0), (i1, s1)) in combinations(enumerate(sess_list), 2)
        ]

    @property
    def n_session(self):
        return len(self._sessions_pairs)

    def __len__(self):
        return self.n_session

    def __repr__(self) -> str:
        return f"{type(self).__name__} object with {self.n_session} sessions"

    @property
    def isempty(self):
        if self._sessions_pairs == [] or self._sessions_pairs is None:
            return True
        else:
            return False

    def __getitem__(self, idx):
        if self.isempty:
            return self

        if isinstance(idx, (int, slice, np.int32, np.int64, np.ndarray)):
            return self._sessions_pairs[idx]
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
        return self._sessions_pairs[index]

    @property
    def distances(self):
        all_distances = [l.distcenters.ravel() for l in self._sessions_pairs]
        return np.concatenate(all_distances)

    @property
    def distances_neighbor(self):
        all_distances = [
            l.distcenters[l.neighbor].ravel() for l in self._sessions_pairs
        ]
        return np.concatenate(all_distances)

    @property
    def correlations(self):
        all_distances = [l.correlations.ravel() for l in self._sessions_pairs]
        return np.concatenate(all_distances)

    @property
    def correlations_neighbor(self):
        all_distances = [
            l.correlations[l.neighbor].ravel() for l in self._sessions_pairs
        ]
        return np.concatenate(all_distances)

    def fit_distance_model(
        self, max_dist: int = 10, n_bins: int = 51, n_bins_out: int = 100
    ):
        dist = self.distances_neighbor
        return fit_center_distribution(
            dist, max_dist=max_dist, n_bins=n_bins, n_bins_out=n_bins_out,
        )

    def fit_correlation_model(
        self, max_dist: int = 10, n_bins: int = 51, n_bins_out: int = 100
    ):
        print(
            "Implement correlation fitting for real, not this function is a place holder and return distance model"
        )
        dist = self.correlations_neighbor
        return fit_center_distribution(
            dist, max_dist=max_dist, n_bins=n_bins, n_bins_out=n_bins_out,
        )

    def get_psame_dist(self, method="distance"):
        if method not in ["distance", "correlation", "best"]:
            raise NotImplementedError(f"Method {method} not implemented")

        if method == "distance":
            (dist_all, dist_same, _, x_est, _, _, _,) = fit_center_distribution(
                self.distances_neighbor
            )
            p_same = calc_psame(dist_same, dist_all)

        elif method == "correlation":
            raise NotImplementedError
        elif method == "best":
            (
                dist_all_dist,
                dist_same_dist,
                _,
                _,
                _,
                error_distance_model,
                _,
            ) = self.fit_distance_model()

            (
                dist_all_corr,
                dist_same_corr,
                _,
                _,
                _,
                error_correlation_model,
                _,
            ) = self.fit_correlation_model

            if error_correlation_model < error_distance_model:
                return calc_psame(dist_all_corr, dist_same_corr)
            else:
                return calc_psame(dist_all_dist, dist_same_dist)

