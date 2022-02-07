from pathlib import Path
import os.path as op
from itertools import combinations
from piCellReg.datatype.Session import SessionList
from piCellReg.datatype.SessionPair import SessionPair
from piCellReg.datatype.SessionPairList import SessionPairList
from piCellReg.utils.fit import fit_center_distribution, calc_psame, psame_matrix

import matplotlib.pyplot as plt
import numpy as np

## List of all sessions pairs
# make a list of sessions
user_path = str(Path.home())

# p1 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201013/stat.npy")
# p2 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201014/stat.npy")
# s_p = SessionPair(Session(p1), Session(p2))
# a = s_p.distcenters
# # s_p.plot_center_distribution()
# s_p.plot_var_distrib(var_to_plot="correlations")

# s_p.plot_join_distrib()


# all_corr = s_p.correlations[s_p.non_nearest_neighbor]
# # all_corr = all_corr[all_corr > 0.2]
# # try to plot the histogram of cells distances
# h = np.histogram(all_corr, bins=np.linspace(0, 1, 100))

# plt.plot(
#     h[1][:-1],
#     h[0],
# )
# plt.show()

#####
######################################################
##  ALL SESSIONS

p_all = op.join(user_path, "Sync/tmpData/crossReg/4466/")
sess_list = SessionList().load_from_s2p(p_all)

# make a list of all SessionPair with all the combinations of Session possible
# L = [
#     SessionPair(s0, s1, id_s0=i0, id_s1=i1)
#     for ((i0, s0), (i1, s1)) in combinations(enumerate(sess_list), 2)
# ]

L = SessionPairList().from_SessionList(sess_list)

# remove bad sessions
# LL = L[[4, 5, 6, 7, 9]]  # [L[l] for l in [4, 5, 6, 7, 9]]
LL = L[[0, 1, 3, 4, 6]]

[l.plot() for l in LL]


# look at distances
all_distances = LL.distances_neighbor
(dist_all, dist_same, dist_different, x_est, _, _, s) = LL.fit_distance_model()


# try to plot the histogram of cells distances
h = np.histogram(all_distances, bins=np.linspace(0, 10, 100), density=True)

plt.plot(h[1][:-1], h[0])
plt.plot(x_est, dist_all)
plt.plot(x_est, dist_same)
plt.plot(x_est, dist_different)
plt.show()


## calculate psame and the psame matrix
p_same, x_est = LL.get_psame_dist()

## output the distance of all the pairs of cells
all_dist = LL.distances
all_psame = psame_matrix(all_dist, p_same, x_est)

# # look at correlations
# all_corr = [l.correlations[l.neighbor].ravel().T for l in LL]
# all_corr = np.concatenate(all_corr)


# # try to plot the histogram of cells distances
# h = np.histogram(all_corr, bins=np.linspace(0, 1, 100))

# plt.plot(
#     h[1][:-1],
#     h[0],
# )
# plt.show
