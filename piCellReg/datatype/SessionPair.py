import matplotlib.pyplot as plt
import numpy as np
from torch import dtype
from piCellReg.datatype.Session import Session
from piCellReg.registration.register import register_image
from piCellReg.registration.utils import shift_image
from piCellReg.utils.helpers import (
    nearest_neighbor_mask,
    neighbor_mask,
    non_nearest_neighbor_mask,
)
from piCellReg.utils.sparse import corr_stack_s, jacquard_s, overlap_s
import bottleneck as bn
import os.path as op


class SessionPair:
    def __init__(
        self,
        s0: Session = None,
        s1: Session = None,
        id_s0: int = 0,
        id_s1: int = 1,
        *,
        max_dist=10,
    ) -> None:
        self._pair_ids = [id_s0, id_s1]
        self._session_0 = s0
        self._session_1 = s1

        self._relative_offsets = np.array([0, 0])  # [y,x] relative offset s0 to s1
        # next offsets are necessary if the relative offset makes
        # negative index for one or the roi in either of the session
        self._offsets_session_0 = np.array([0, 0])  # [y,x] numpy format
        self._offsets_session_1 = np.array([0, 0])  # [y,x] numpy format

        self._Lx_corrected = bn.nanmax([self._session_0.Lx, self._session_1.Lx])
        self._Ly_corrected = bn.nanmax([self._session_0.Ly, self._session_1.Ly])

        self._rotation = 0
        self._dist_centers = None
        self._correlation = None
        self._overlaps = None
        self._jacquard = None
        self._max_dist = max_dist

        # do it by default for now
        self._calc_offset()

    @property
    def cell_ids(self):
        sz = np.array([self._session_0.n_cells, self._session_1.n_cells])
        s_0_sess_id = np.ones(sz, dtype=np.int32) * self._session_0._idx
        s_0_cells_id = (
            np.arange(sz[0], dtype=np.int32)[:, None]
            * np.ones(sz[1], dtype=np.int32)[None, :]
        )

        s_1_sess_id = np.ones(sz, dtype=np.int32) * self._session_1._idx
        s_1_cells_id = (
            np.ones(sz[0], dtype=np.int32)[:, None]
            * np.arange(sz[1], dtype=np.int32)[None, :]
        )

        return s_0_sess_id, s_0_cells_id, s_1_sess_id, s_1_cells_id

    @property
    def cell_ids_lin(self):
        s_0_sess_id, s_0_cells_id, s_1_sess_id, s_1_cells_id = self.cell_ids

        return (
            s_0_sess_id.ravel(),
            s_0_cells_id.ravel(),
            s_1_sess_id.ravel(),
            s_1_cells_id.ravel(),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__} object with sessions {self._pair_ids[0]} and {self._pair_ids[1]}"

    def _calc_offset(self, do_rotation=False):
        if do_rotation:
            self._relative_offsets, self._rotation = register_image(
                self._session_0._mean_image_e,
                self._session_1._mean_image_e,
                do_rotation=True,
            )
        else:
            self._relative_offsets = register_image(
                self._session_0._mean_image_e, self._session_1._mean_image_e
            )

        # as we take consider the session 1 as the "moving image"
        self._offsets_session_1 = self._relative_offsets

        # Check if the shift make some of the coordinates/ pixel indexes to
        # be out of the image range or negative
        _, y_pix_corr, x_pix_corr = self._session_1.get_coordinates(
            x_offset=self._offsets_session_1[1],
            y_offset=self._offsets_session_1[0],
            rotation=self._rotation,
        )

        min_y = bn.nanmin(y_pix_corr)
        max_y = bn.nanmax(y_pix_corr)

        min_x = bn.nanmin(x_pix_corr)
        max_x = bn.nanmax(x_pix_corr)

        if min_y < 0:
            # fix the y offset if negative
            self._offsets_session_0[0] += abs(min_y)
            self._offsets_session_1[0] += abs(min_y)
            max_y += abs(min_y)

        if min_x < 0:
            # fix the y offset if negative
            self._offsets_session_0[1] += abs(min_x)
            self._offsets_session_1[1] += abs(min_x)
            max_x += abs(min_x)

        if max_y > self._Ly_corrected:
            # fix the y range if larger than the initial one
            self._Ly_corrected = max_y
        if max_x > self._Lx_corrected:
            # fix the y range if larger than the initial one
            self._Lx_corrected = max_x

    @property
    def distcenters(self):
        if all(self._relative_offsets == 0):
            # make sure we have the offsets done
            self._calc_offset()

        if self._dist_centers is None:
            # calculate the distance between all the pairs of cells between two sessions
            x_dists = self._session_0._x_center[:, None] - (
                self._session_1._x_center[None, :] + self._relative_offsets[1]
            )
            y_dists = self._session_0._y_center[:, None] - (
                self._session_1._y_center[None, :] + self._relative_offsets[0]
            )
            # calculate distance between all the pairs of cells
            self._dist_centers = np.sqrt(x_dists ** 2 + y_dists ** 2)
        return self._dist_centers

    @property
    def overlaps(self):
        if all(self._relative_offsets == 0):
            # make sure we have the offsets done
            self._calc_offset()

        if self._overlaps is None:
            hm0 = self._session_0.to_sparse_hot_mat(
                x_offset=self._offsets_session_0[1],
                y_offset=self._offsets_session_0[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )
            hm1 = self._session_1.to_sparse_hot_mat(
                x_offset=self._offsets_session_1[1],
                y_offset=self._offsets_session_1[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )

            self._overlaps = overlap_s(hm0, hm1)

        return self._overlaps

    @property
    def correlations(self):
        if all(self._relative_offsets == 0):
            # make sure we have the offsets done
            self._calc_offset()

        if self._correlation is None:
            lm0 = self._session_0.to_sparse_lam_mat(
                x_offset=self._offsets_session_0[1],
                y_offset=self._offsets_session_0[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )
            lm1 = self._session_1.to_sparse_lam_mat(
                x_offset=self._offsets_session_1[1],
                y_offset=self._offsets_session_1[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )
            self._correlation = corr_stack_s(lm0, lm1)

        return self._correlation

    @property
    def jacquard(self):
        if all(self._relative_offsets == 0):
            self._calc_offset()

        if self._jacquard is None:
            hm0 = self._session_0.to_sparse_hot_mat(
                x_offset=self._offsets_session_0[1],
                y_offset=self._offsets_session_0[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )
            hm1 = self._session_1.to_sparse_hot_mat(
                x_offset=self._offsets_session_1[1],
                y_offset=self._offsets_session_1[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )

            self._jacquard = jacquard_s(hm0, hm1)

        return self._jacquard

    @property
    def iscell(self):
        return (
            self._session_0._iscell[:, None] * self._session_1._iscell[None, :]
        ) == 1

    @property
    def nearest_neighbor(self):
        m = nearest_neighbor_mask(self.distcenters)
        return np.logical_and(m, self.iscell)

    @property
    def neighbor(self):
        m = neighbor_mask(self.distcenters, radius=self._max_dist)
        return np.logical_and(m, self.iscell)

    @property
    def non_nearest_neighbor(self):
        m = non_nearest_neighbor_mask(self.distcenters, radius=self._max_dist)
        return np.logical_and(m, self.iscell)

    def plot(self):

        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(self._session_0._mean_image_e, cmap="Blues")
        s1_s = shift_image(self._session_1._mean_image_e, -self._relative_offsets)

        plt.imshow(s1_s, alpha=0.5, cmap="Greens")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(
            self._session_0.get_projection(
                x_offset=self._offsets_session_0[1],
                y_offset=self._offsets_session_0[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            ),
            cmap="Blues",
            interpolation="nearest",
        )
        plt.imshow(
            self._session_1.get_projection(
                x_offset=self._offsets_session_1[1],
                y_offset=self._offsets_session_1[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            ),
            alpha=0.5,
            cmap="Greens",
            interpolation="nearest",
        )
        plt.axis("off")

        plt.show()

    def plot_center_distribution(self, n_bins=None):

        if n_bins is None:
            n_bins = (self._max_dist * 2) + 1

        x_dists = self._session_0._x_center[:, None] - (
            self._session_1._x_center[None, :] + self._relative_offsets[1]
        )
        y_dists = self._session_0._y_center[:, None] - (
            self._session_1._y_center[None, :] + self._relative_offsets[0]
        )

        edges = np.linspace(-self._max_dist, self._max_dist, n_bins)
        H, _, _ = np.histogram2d(x_dists.ravel(), y_dists.ravel(), bins=(edges, edges))

        t = np.array([0, (n_bins - 1) / 2, n_bins - 1])
        centers = [-self._max_dist, 0, self._max_dist]

        plt.imshow(H)

        plt.xlabel(" x Displacement")
        plt.ylabel(" y Displacement")

        plt.xticks(t, centers)
        plt.yticks(t, centers)
        plt.show()

    def plot_var_distrib(self, var_to_plot="correlations", n_bins=None, save_path=None):

        if var_to_plot not in ["correlations", "overlaps", "jacquard"]:
            raise ValueError(f"variable {var_to_plot} not fount")

        if n_bins is None:
            n_bins = (self._max_dist * 2) + 1

        x_dists = self._session_0._x_center[:, None] - (
            self._session_1._x_center[None, :] + self._relative_offsets[1]
        )
        y_dists = self._session_0._y_center[:, None] - (
            self._session_1._y_center[None, :] + self._relative_offsets[0]
        )

        mask = self.neighbor.ravel()

        if var_to_plot == "correlations":
            weigth = self.correlations.ravel()
        elif var_to_plot == "overlaps":
            weigth = self.overlaps.ravel()
        elif var_to_plot == "jacquard":
            weigth = self.jacquard.ravel()

        edges = np.linspace(-self._max_dist, self._max_dist, n_bins)

        H_count, _, _ = np.histogram2d(
            x_dists.ravel()[mask], y_dists.ravel()[mask], bins=(edges, edges)
        )

        H, _, _ = np.histogram2d(
            x_dists.ravel()[mask],
            y_dists.ravel()[mask],
            bins=(edges, edges),
            weights=weigth[mask],
        )

        H = H / H_count

        t = np.array([0, (n_bins - 1) / 2, n_bins - 1])
        centers = [-self._max_dist, 0, self._max_dist]

        plt.imshow(H)

        plt.xlabel(" x Displacement")
        plt.ylabel(" y Displacement")

        plt.xticks(t, centers)
        plt.yticks(t, centers)

        plt.title(f"Average {var_to_plot}")

        plt.colorbar()
        plt.show()

        if save_path is not None:
            namefile = f"Session-{self._pair_ids[0]}-vs-Session{self._pair_ids[1]}-average{var_to_plot}"
            plt.savefig(op.join(save_path, namefile), format="png")
            plt.close()

    def plot_join_distrib(self, n_bins=30):

        d = self.distcenters[self.neighbor]
        c = self.correlations[self.neighbor]

        edges_dist = np.linspace(0, self._max_dist, n_bins)
        edges_corr = np.linspace(0, 1, n_bins)

        H, _, _ = np.histogram2d(d.ravel(), c.ravel(), bins=(edges_dist, edges_corr),)

        t = np.linspace(0, n_bins - 1, 5)
        center_dist = np.linspace(0, self._max_dist, len(t))
        center_corr = np.linspace(0, 1, len(t))

        plt.imshow(H)

        plt.xlabel(" centroid distance")
        plt.ylabel(" correlation ")

        plt.xticks(t, center_dist)
        plt.yticks(t, center_corr)

        plt.show()
