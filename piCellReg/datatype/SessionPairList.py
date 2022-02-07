from dataclasses import dataclass
import itertools
from piCellReg.datatype.SessionPair import SessionPair
import numpy as np
from itertools import combinations
from piCellReg.utils.fit import fit_center_distribution, calc_psame, psame_matrix
import matplotlib.pyplot as plt
import bottleneck as bn


@dataclass
class SessionPairList:
    """
    Wrapper to group SessionsPairs
    """

    # general infos
    _sessions_pairs = []
    _max_dist = 10
    _max_n_cell = None

    # load all available sessions in a root path
    def from_list(self, session_pairs_list: list, max_dist=10):
        self._sessions_pairs = session_pairs_list
        self._max_dist = max_dist
        return self

    def from_SessionList(self, sess_list, max_dist=10):
        self._sessions_pairs = [
            SessionPair(s0, s1, id_s0=i0, id_s1=i1, max_dist=max_dist)
            for ((i0, s0), (i1, s1)) in combinations(enumerate(sess_list), 2)
        ]
        return self

    @property
    def max_n_cell(self):
        """
        max_n_cell return the maximum number of cells of the biggest session
        in the session pair list 
        """
        if self._max_n_cell is None:
            m = [
                bn.nanmax([s._session_0.n_cells, s._session_1.n_cells])
                for s in self._sessions_pairs
            ]
            self._max_n_cell = bn.nanmax(m)
        else:
            return self._max_n_cell

    @property
    def n_session(self):
        """
        n_session returns the number of unique sessions
        """
        all_ids = [i._pair_ids for i in self]
        all_ids = np.concatenate(all_ids)
        return len(self.session_ids)

    @property
    def session_ids(self):
        """
        n_session returns ids of unique sessions
        """
        all_ids = [i._pair_ids for i in self]
        return np.unique(np.concatenate(all_ids))

    @property
    def n_pairs(self):
        return len(self._sessions_pairs)

    def __len__(self):
        return self.n_pairs

    def __repr__(self) -> str:
        return f"{type(self).__name__} object with {self.n_session} pairs of sessions"

    @property
    def isempty(self):
        if self._sessions_pairs == [] or self._sessions_pairs is None:
            return True
        else:
            return False

    def __getitem__(self, idx):
        if self.isempty:
            return self

        if isinstance(idx, (int, slice, np.int32, np.int64)):
            return self._sessions_pairs[idx]

        if isinstance(idx, (list)):
            print(type(self))
            return type(self)().from_list(
                [self._sessions_pairs[i] for i in idx], max_dist=self.max_dist,
            )
        else:
            raise TypeError(f"Unsupported indexing type {type(idx)}")

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        index = self._index
        if index > len(self) - 1:
            raise StopIteration

        self._index += 1
        return self._sessions_pairs[index]

    @property
    def max_dist(self):
        return self._max_dist

    @max_dist.setter
    def max_dist(self, val):
        self._max_dist = val
        # and update the max dist inside the list
        for l in self._sessions_pairs:
            l._max_dist = val

    @property
    def all_cell_ids(self):
        s_0_sess_id = []
        s_0_cells_id = []
        s_1_sess_id = []
        s_1_cells_id = []

        for p in self:
            a, b, c, d = p.cell_ids_lin
            s_0_sess_id.append(a)
            s_0_cells_id.append(b)
            s_1_sess_id.append(c)
            s_1_cells_id.append(d)

        return (
            np.concatenate(s_0_sess_id),
            np.concatenate(s_0_cells_id),
            np.concatenate(s_1_sess_id),
            np.concatenate(s_1_cells_id),
        )

    @property
    def all_cell_lin_ids(self):
        """
        all_cell_lin_ids 
        this code convert cell indexes from (cell_id, session_id) -> unique cell id
        """
        s_0_sess_id, s_0_cells_id, s_1_sess_id, s_1_cells_id = self.all_cell_ids
        sz = [self.n_session, self.max_n_cell]
        s_0_cells_id_lin = np.ravel_multi_index([s_0_sess_id, s_0_cells_id], sz)
        s_1_cells_id_lin = np.ravel_multi_index([s_1_sess_id, s_1_cells_id], sz)

        return s_0_cells_id_lin, s_1_cells_id_lin

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
            return calc_psame(dist_same, dist_all), x_est

        elif method == "correlation":
            raise NotImplementedError
        elif method == "best":
            (
                dist_all_dist,
                dist_same_dist,
                _,
                x_est,
                _,
                error_distance_model,
                _,
            ) = self.fit_distance_model()

            (
                dist_all_corr,
                dist_same_corr,
                _,
                x_est,
                _,
                error_correlation_model,
                _,
            ) = self.fit_correlation_model

            if error_correlation_model < error_distance_model:
                return calc_psame(dist_all_corr, dist_same_corr), x_est
            else:
                return calc_psame(dist_all_dist, dist_same_dist), x_est

    def plot_join_distrib(self, n_bins=30):

        d = self.distances_neighbor
        c = self.correlations_neighbor

        edges_dist = np.linspace(0, self.max_dist, n_bins)
        edges_corr = np.linspace(0, 1, n_bins)

        H, _, _ = np.histogram2d(d.ravel(), c.ravel(), bins=(edges_dist, edges_corr),)

        t = np.linspace(0, n_bins - 1, 5)
        center_dist = np.linspace(0, self.max_dist, len(t))
        center_corr = np.linspace(0, 1, len(t))

        plt.imshow(H)

        plt.xlabel(" centroid distance")
        plt.ylabel(" correlation ")

        plt.xticks(t, center_dist)
        plt.yticks(t, center_corr)

        plt.show()
