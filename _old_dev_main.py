from piCellReg.datatype.SessionList import SessionList
from piCellReg.datatype.Session import Session
from piCellReg.registration.utils import shift_image
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
import numpy as np
from scipy import sparse
from piCellReg.utils.sparse import jacquard_s, overlap_s, corr_stack_s
from piCellReg.utils.fit import fit_center_distribution
import bottleneck as bn
from pathlib import Path
import os.path as op

user_path = str(Path.home())

p1 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201013/stat.npy")
p2 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201014/stat.npy")

s1 = Session(p1)
s2 = Session(p2)

###
p_all = op.join(user_path, "Sync/tmpData/crossReg/4466/")
s_all = Aln(p_all)
###
#
offset = phase_cross_correlation(
    s1._mean_image_e, s2._mean_image_e, upsample_factor=100
)
im_s = shift_image(s2._mean_image_e, -offset[0])

m1 = s1._iscell.astype(bool)
m2 = s2._iscell.astype(bool)

plt.figure(figsize=(20, 20))
plt.subplot(221)
plt.imshow(s1._mean_image_e)
plt.plot(s1._x_center[m1], s1._y_center[m1], "+r", ms=2)


plt.subplot(222)
plt.imshow(s2._mean_image_e)
plt.plot(s2._x_center[m2], s2._y_center[m2], "+r", ms=2)


plt.subplot(223)
# plt.imshow(s1._mean_image_e, cmap="Reds")
# plt.imshow(s2._mean_image_e, alpha=0.5, cmap="Greens")
plt.plot(s1._x_center[m1], s1._y_center[m1], "+r", ms=2)
# plt.imshow(im_s )
plt.plot(s2._x_center[m2] + offset[0][1], s2._y_center[m2] + offset[0][0], "+k", ms=2)

plt.subplot(224)
plt.imshow(s1._mean_image_e)
plt.imshow(im_s, alpha=0.5, cmap="magma")

plt.show()

# try overlap
hm1 = s1.to_hot_mat()
hm1 = hm1.reshape((hm1.shape[0], -1))

hm2 = s2.to_hot_mat(x_shift=-offset[0][1], y_shift=-offset[0][0])
hm2 = hm2.reshape((hm2.shape[0], -1))

h1 = sparse.csr_matrix(hm1, dtype=np.int8)
h2 = sparse.csr_matrix(hm2, dtype=np.int8)

lm1 = s1.to_lam_mat()
lm1 = lm1.reshape((lm1.shape[0], -1))

lm2 = s2.to_lam_mat(x_shift=-offset[0][1], y_shift=-offset[0][0])
lm2 = lm2.reshape((lm2.shape[0], -1))

l1 = sparse.csr_matrix(lm1, dtype=np.float64)
l2 = sparse.csr_matrix(lm2, dtype=np.float64)

# calculate the distance between all the pairs of cells between two sessions
x_dists = s1._x_center[:, None] - (s2._x_center[None, :] + offset[0][1])
y_dists = s1._y_center[:, None] - (s2._y_center[None, :] + offset[0][0])

# calculate distance between all the pairs of cells
dists = np.sqrt(x_dists ** 2 + y_dists ** 2)
N_dist = dists[dists < 14].ravel()

# calculate all the overlap of the footprints
ov_sp = overlap_s(h1, h2)
# calculate all the jacquard distance of the footprints
J = jacquard_s(h1, h2)
# calculate all the cross correlations of the footprints
C = corr_stack_s(l1, l2)

# store a vector of distance and correlation of only the nearest neighbor
NN_idx = bn.nanargmin(dists, axis=1)

# do the same for the other neighbors (dists and correlations )
NNN_m = dists < 14
NNN_m[(np.arange(NNN_m.shape[0]), NN_idx)] = False
NNN_idx = np.nonzero(NNN_m)

# fit the distribution
dist_all, dist_same, dist_different, _, _, _ = fit_center_distribution(N_dist)

# calculate psame
def calc_psame(dist_same, dist_all):
    # find p_same knowing the distance/corr
    p_same = dist_same / dist_all
    # here we should may be fit a sigmoid distribution they do a little
    # trick in their code by setting the first value of the distribution to the second value
    # see line 73 of compute_centroid_distances_model
    # this could be explained by the fact that the observed distribution is lognormal due to
    # inacuracy in the alignment of the two image (we will never or rarely have a perfect alignment,
    #  thus a drop at zero and a lognormal distribution ). However the real underlying distribution should
    # be sigmoidal. This could justify a sigmoid fit here
    # eg we have P_same _obs (lognormal) we want p_same_expected (sigmoid)
    return p_same


def psame_matrix(dist_mat, p_same_dist, p_same_centers):

    sz = dist_mat.shape
    # linearise matrix for simplicity
    d = dist_mat.ravel()
    # keep values inside of the p_same centers range
    mask = np.logical_and(d >= p_same_centers[0], d <= p_same_centers[-1])
    # find closest p_same center fo each dist_mat values
    i = bn.nanargmin(abs(d[mask, None] - p_same_centers[None, :]), axis=1)

    # output
    out = np.empty_like(d)
    out.fill(np.nan)

    # affect p_same value corresponding to the closest center in the distribution
    # and reshape the output to the initial shape
    out[mask] = p_same_dist[i]
    out = out.reshape(sz)
    return out


## calculate psame and the psame matrix
p_same = calc_psame(dist_same, dist_all)
p_same_centers = np.linspace(0, 15, 100)  # it should be outputed from the fit function

putative_same = psame_matrix(dist, p_same, p_same_centers)

plt.plot()

# for assignment of psame.look at
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
# hungarian algorythm
