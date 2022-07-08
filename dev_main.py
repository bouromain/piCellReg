from pathlib import Path

import os.path as op
from itertools import combinations
from piCellReg.datatype.Session import SessionList
from piCellReg.datatype.SessionPair import SessionPair
from piCellReg.datatype.SessionPairList import SessionPairList
from piCellReg.utils.fit import fit_center_distribution, calc_psame, psame_matrix
from piCellReg.utils.clustering import cluster_cell, find_duplicated_nodes

import matplotlib.pyplot as plt
import numpy as np
import bottleneck as bn

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
LL = L[[0, 1, 3, 4, 5]]

# [l.plot() for l in LL]


# look at distances
all_distances = LL.distances[LL.neighbors]
(dist_all, dist_same, dist_different, x_est, _, _, s) = LL.fit_distance_model()

## calculate psame and the psame matrix
p_same, x_est = LL.get_psame_dist()

# try to plot the histogram of cells distances
h = np.histogram(all_distances, bins=np.linspace(0, 10, 100), density=True)

# plt.plot(h[1][:-1], h[0])
# plt.plot(x_est, dist_all)
# plt.plot(x_est, dist_same)
# plt.plot(x_est, dist_different)
# x_ps = bn.nanmax(np.nonzero(p_same>0.95))
# plt.plot([ x_est[x_ps] , x_est[x_ps]], [0,0.5],"-r")
# plt.ylim([0,0.35])
# plt.show()

## output the distance of all the pairs of cells
all_dist = LL.distances
all_psame = psame_matrix(all_dist, p_same, x_est)
putative_same_mask = np.logical_and(all_psame > 0.05, LL.neighbors)

##### FOR DEV
edges_list = LL.all_cell_lin_ids
edges_list = np.vstack(edges_list)
edges_list = edges_list[:, putative_same_mask]

edge_corr = LL.correlations
edge_corr = edge_corr[putative_same_mask]

sess_vertex = LL.all_cell_lin_sess_id
sess_vertex = {s[0]: s[1] for s in sess_vertex.T}

# c = cluster_cell(edges_list, edge_corr, sess_vertex)

import numpy as np
import networkx as nx
from itertools import product

edgesList = edges_list.T
weigths = edge_corr
session_list = sess_vertex

if edgesList.shape[0] != weigths.shape[0]:
    raise ValueError(
        "Variable size mismatch,edge list and weights should share their first dimension"
    )
weigthed_edges_list = [(v[0], v[1], w) for v, w in zip(edgesList, weigths)]
big_G = nx.Graph()
big_G.add_weighted_edges_from(weigthed_edges_list)
# set a "session" attribute
nx.set_node_attributes(big_G, session_list, "session")

# find node that are duplicate (one cluster has several putative candidate in a session)
no_duplicate_idx, duplicates_idx = find_duplicated_nodes(big_G)


##
c = 0
G = big_G

list_clusters = [c for c in nx.connected_components(G)]
for it_clust in list_clusters:
    # take subgraph with only nodes from one cluster
    tmp_clust = G.subgraph(it_clust)

    # now find if we have duplicate node in a layer
    tmp_sess_list = np.array([v["session"] for _, v in tmp_clust.nodes(data=True)])
    tmp_node_id = it_clust
    if len(set(tmp_sess_list)) < len(tmp_sess_list):
        c += 1

# see
# https://www.francescobonchi.com/CCtuto_kdd14.pdf
# interesting 2 7 8

aa = G.subgraph(list_clusters[8])
sessions = {lab: v["session"] for lab, v in aa.nodes(data=True)}
nx.draw(aa, labels=sessions, with_labels=True)  #
plt.show()


##
F = nx.fiedler_vector(aa)
plt.plot(F)
plt.show()

n1 = np.array(aa.nodes())[F > 0]
n2 = np.array(aa.nodes())[F < 0]

g1 = aa.subgraph(n1)
g2 = aa.subgraph(n2)

plt.subplot(121)
nx.draw(g1, with_labels=True)
plt.subplot(122)
nx.draw(g2, with_labels=True)
plt.show()

#### END FOR DEV

LL[0].plot_same_cell(0, 32)
plt.subplot(121)
plt.imshow(LL[0]._session_0.get_roi(0, margin=20))
plt.subplot(122)
plt.imshow(LL[0]._session_1.get_roi(32, margin=20))
plt.show()
