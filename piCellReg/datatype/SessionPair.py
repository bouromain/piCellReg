from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from piCellReg.datatype.Aln import Aln
from piCellReg.datatype.Session import Session
from piCellReg.registration.register import register_image
from piCellReg.registration.utils import shift_image
from piCellReg.utils.helpers import (
    nearest_neighbor_mask,
    neighbor_mask,
    non_nearest_neighbor_mask,
)
from piCellReg.utils.sparse import corr_stack_s, jacquard_s, overlap_s


def _calc_dist(x_0, y_0, x_1, y_1, offset):
    # calculate the distance between all the pairs of cells between two sessions
    x_dists = x_0[:, None] - (x_1[None, :] + offset[1])
    y_dists = y_0[:, None] - (y_1[None, :] + offset[0])

    # calculate distance between all the pairs of cells
    return np.sqrt(x_dists ** 2 + y_dists ** 2)


class SessionPair:
    def __init__(
        self, s0: Session = None, s1: Session = None, id_s0: int = 0, id_s1: int = 1
    ) -> None:
        self._pair_ids = [id_s0, id_s1]
        self._session_0 = s0
        self._session_1 = s1

        self._relative_offsets = None  # relative offset s0 to s1
        # next offsets are necessary if the relative offset makes
        # negative index for one or the roi in either of the session
        self._offsets_session_0 = None
        self._offsets_session_1 = None

        self._rotation = 0
        self._dist_centers = None
        self._corr = None
        self._overlaps = None
        self._max_dist = 14

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

        # compute the absolute offset for each session
        # a little trick is to always use a positive offset.
        # if any(self._relative_offsets < 0):
        #     ...

    def distcenters(self):
        if self._relative_offsets is None:
            # make sure we have the offsets done
            self._calc_offset()

        if self._dist_centers is None:
            self._dist_centers = _calc_dist(
                self._session_0._x_center,
                self._session_0._y_center,
                self._session_1._x_center,
                self._session_1._y_center,
                self._relative_offsets,
            )
        return self._dist_centers

    def overlaps(self):
        if self._relative_offsets is None:
            # make sure we have the offsets done
            self._calc_offset()

        if self._overlaps is None:
            hm0 = self._session_0.to_sparse_hot_mat()
            hm1 = self._session_1.to_sparse_hot_mat(
                x_shift=-self._relative_offsets[1], y_shift=-self._relative_offsets[0]
            )

            self._overlaps = overlap_s(hm0, hm1)

            return self._overlaps

    def correlations(self):
        if self._relative_offsets is None:
            # make sure we have the offsets done
            self._calc_offset()

        if self._corr is None:
            lm0 = self._session_0.to_sparse_lam_mat()
            lm1 = self._session_1.to_sparse_lam_mat(
                x_shift=-self._relative_offsets[1], y_shift=-self._relative_offsets[0]
            )
            self._corr = corr_stack_s(lm0, lm1)

            return self._corr

    @property
    def nearest_neighbor(self):
        return nearest_neighbor_mask(self.distcenters())

    @property
    def neighbor_mask(self):
        return neighbor_mask(self.distcenters(), radius=self._max_dist)

    @property
    def non_nearest_neighbor_mask(self):
        return non_nearest_neighbor_mask(self.distcenters(), radius=self._max_dist)

    def plot(self):
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(self._session_0._mean_image_e, cmap="Reds")
        s1_s = shift_image(self._session_1._mean_image_e, -self._relative_offsets)

        plt.imshow(s1_s, alpha=0.5, cmap="Greens")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(
            self._session_0.get_projection(), cmap="Reds", interpolation="nearest"
        )
        plt.imshow(
            self._session_1.get_projection(
                x_shift=self._relative_offsets[1],
                y_shift=self._relative_offsets[0],
                theta=self._rotation,
            ),
            alpha=0.5,
            cmap="Greens",
            interpolation="nearest",
        )
        plt.axis("off")

        plt.show()
