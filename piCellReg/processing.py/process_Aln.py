import numpy as np
from itertools import combinations
from piCellReg.datatype.Aln import Aln
from piCellReg.registration.register import register_image

##
import os.path as op
from pathlib import Path


user_path = str(Path.home())

p_all = op.join(user_path, "Sync/tmpData/crossReg/4466/")
all_sess = Aln(p_all)


def _calc_dist(x_0, y_0, x_1, y_1, offset):
    # calculate the distance between all the pairs of cells between two sessions
    x_dists = x_0[:, None] - (x_1[None, :] + offset[1])
    y_dists = y_0[:, None] - (y_1[None, :] + offset[0])

    # calculate distance between all the pairs of cells
    return np.sqrt(x_dists ** 2 + y_dists ** 2)


# def calc_dist(all_sess:Aln,max_dist: int =14):
max_dist = 14

for (s0, s1) in combinations(enumerate(a), 2):
    offsets = register_image(s0._mean_image_e, s1._mean_image_e)
    dists = _calc_dist(s0._x_center, s0._y_center, s1._x_center, s1._y_center, offsets)

