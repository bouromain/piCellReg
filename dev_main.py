from pathlib import Path
import os.path as op
from itertools import combinations
from piCellReg.datatype.Session import SessionList, Session
from piCellReg.datatype.SessionPair import SessionPair
from piCellReg.utils.fit import fit_center_distribution

import bottleneck as bn

import matplotlib.pyplot as plt
import numpy as np

## List of all sessions pairs
# make a list of sessions
user_path = str(Path.home())

p1 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201013/stat.npy")
p2 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201014/stat.npy")
s_p = SessionPair(Session(p1), Session(p2))
a = s_p.distcenters
# s_p.plot_center_distribution()
s_p.plot_var_distrib(var_to_plot="correlations")


all_corr = s_p.correlations[s_p.neighbor]
# all_corr = all_corr[all_corr > 0.2]
# try to plot the histogram of cells distances
h = np.histogram(all_corr, bins=np.linspace(0, 1, 100))

plt.plot(
    h[1][:-1], np.log(h[0]),
)
plt.show()

######################################################
##  ALL SESSIONS

p_all = op.join(user_path, "Sync/tmpData/crossReg/4466/")
sess_list = SessionList().load_from_s2p(p_all)

# make a list of all SessionPair with all the combinations of Session possible
L = [
    SessionPair(s0, s1, id_s0=i0, id_s1=i1)
    for ((i0, s0), (i1, s1)) in combinations(enumerate(sess_list), 2)
]
# remove shitty sessions
LL = [L[l] for l in [0, 1, 3, 4, 6]]

# look at distances
all_distances = [l.distcenters[l.neighbor].ravel() for l in LL]
all_distances = np.concatenate(all_distances)

(dist_all, dist_same, dist_different, _, _, _, _) = fit_center_distribution(
    all_distances
)

plt.plot(dist_all)
plt.show()

# look at correlations
all_corr = [l.correlations[l.neighbor].ravel().T for l in LL]
all_corr = np.concatenate(all_corr)


# try to plot the histogram of cells distances
h = np.histogram(all_corr, bins=np.linspace(0, 1, 100))

plt.plot(
    h[1][:-1], h[0],
)
plt.show()

