from piCellReg.datatype.Aln import Aln
from piCellReg.datatype.Session import Session
from piCellReg.registration.utils import shift_image
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
import numpy as np
from scipy import sparse
from piCellReg.utils.sparse import jacquard_s, overlap_s, corr_stack_s
import bottleneck as bn

p1 = "/Users/bouromain/Sync/tmpData/crossReg/4466/20201013/stat.npy"
p2 = "/Users/bouromain/Sync/tmpData/crossReg/4466/20201014/stat.npy"

s1 = Session(p1)
s2 = Session(p2)

###
s_all = Aln("/Users/bouromain/Sync/tmpData/crossReg/4466/")
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

# calculate the distance between all the pairs of cells inbetween two sessions
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

# store a vector of distance and correlation of only the nearest neibour
NN_idx = bn.nanargmin(dists, axis=1)

# do the same for the other neighbors (dists and correlations )
NNN_m = dists < 14
NNN_m[(np.arange(NNN_m.shape[0]), NN_idx)] = False
NNN_idx = np.nonzero(NNN_m)


# for the fit see
# scipy optimise curve fit
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

from scipy.optimize import curve_fit


def logifunc(x, A, x0, k, off):
    return A / (1 + np.exp(-k * (x - x0))) + off


def lognormal(x, B, mu, sigma):
    """pdf of lognormal distribution"""

    return (
        B
        * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
        / (x * sigma * np.sqrt(2 * np.pi))
    )


def fit_func(x, A, x0, k, off, B, mu, sigma):
    return logifunc(x, A, x0, k, off) + lognormal(x, B, mu, sigma)


edges = np.linspace(0.001, 14, 51)
binned, _ = np.histogram(N_dist, edges, density=True)
binned = binned + np.finfo(binned.dtype).eps
centers = np.linspace(0.001, 14, 50)

param_bounds = ([0, 0, 0, 0, 0, 0.0001, 0], [1, np.inf, np.inf, np.inf, 1, 2, 4])
sol, err = curve_fit(fit_func, centers, binned, bounds=param_bounds)
print(sol)

# implement weigth of each distribution

x = np.linspace(0, 14, 100)
y = logifunc(x, *sol[:4])
y2 = lognormal(x, *sol[4:])
y3 = fit_func(x, *sol)

plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(centers, binned)
plt.show()


# calculate psame (Not sure I understood...)

# for assignment of psame.look at
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
# hungarian algorythm

