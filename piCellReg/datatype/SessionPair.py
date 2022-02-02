import matplotlib.pyplot as plt
import numpy as np
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


class SessionPair:
    def __init__(
        self, s0: Session = None, s1: Session = None, id_s0: int = 0, id_s1: int = 1
    ) -> None:
        self._pair_ids = [id_s0, id_s1]
        self._session_0 = s0
        self._session_1 = s1

        self._relative_offsets = [0, 0]  # [y,x] relative offset s0 to s1
        # next offsets are necessary if the relative offset makes
        # negative index for one or the roi in either of the session
        self._offsets_session_0 = [0, 0]  # [y,x] numpy format
        self._offsets_session_1 = [0, 0]  # [y,x] numpy format

        self._Lx_corrected = bn.nanmax([self._session_0.Lx, self._session_1.Lx])
        self._Ly_corrected = bn.nanmax([self._session_0.Ly, self._session_1.Ly])

        self._rotation = 0
        self._dist_centers = None
        self._corr = None
        self._overlaps = None
        self._jacquard = None
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

    def distcenters(self):
        if self._relative_offsets is None:
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

    def overlaps(self):
        if self._relative_offsets is None:
            # make sure we have the offsets done
            self._calc_offset()

        if self._overlaps is None:
            hm0 = self._session_0.to_sparse_hot_mat(
                x_shift=-self._offsets_session_0[1],
                y_shift=-self._offsets_session_0[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )
            hm1 = self._session_1.to_sparse_hot_mat(
                x_shift=-self._offsets_session_1[1],
                y_shift=-self._offsets_session_1[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )

            self._overlaps = overlap_s(hm0, hm1)

            return self._overlaps

    def correlations(self):
        if self._relative_offsets is None:
            # make sure we have the offsets done
            self._calc_offset()

        if self._corr is None:
            lm0 = self._session_0.to_sparse_lam_mat(
                x_shift=-self._offsets_session_0[1],
                y_shift=-self._offsets_session_0[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )
            lm1 = self._session_1.to_sparse_lam_mat(
                x_shift=-self._offsets_session_1[1],
                y_shift=-self._offsets_session_1[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )
            self._corr = corr_stack_s(lm0, lm1)

            return self._corr

    def jacquard(self):
        if self._relative_offsets is None:
            self._calc_offset()

        if self._jacquard is None:
            hm0 = self._session_0.to_sparse_hot_mat(
                x_shift=-self._offsets_session_0[1],
                y_shift=-self._offsets_session_0[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )
            hm1 = self._session_1.to_sparse_hot_mat(
                x_shift=-self._offsets_session_1[1],
                y_shift=-self._offsets_session_1[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            )

            self._jacquard = jacquard_s(hm0, hm1)

            return self._jacquard

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
            self._session_0.get_projection(
                x_shift=-self._offsets_session_0[1],
                y_shift=-self._offsets_session_0[0],
                rotation=self._rotation,
                L_x=self._Lx_corrected,
                L_y=self._Ly_corrected,
            ),
            cmap="Reds",
            interpolation="nearest",
        )
        plt.imshow(
            self._session_1.get_projection(
                x_shift=-self._offsets_session_1[1],
                y_shift=-self._offsets_session_1[0],
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
