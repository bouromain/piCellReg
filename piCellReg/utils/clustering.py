import numpy as np
import networkx as nx
from itertools import product


def cluster_cell(edgesList: np.ndarray, weigths: np.ndarray, session_list: list):
    """
    cluster_cell [summary]

    Parameters
    ----------
    edgesList : np.ndarray
        list of edges  [[Vertex 1, Vertex 2], ... , [Vertex m, Vertex n]] of size [2, n_edges]
    weights : np.ndarray
        weights corresponding to the edges list
    session_list : dict
        sessions id corresponding to the each vertex {vertex1:1, vertex2:2, vertex3:3, vertex4:3}

    Raises
    ------
    ValueError
        [description]
    """
    ...
    if edgesList.shape[0] != weigths.shape[0]:
        raise ValueError(
            "Variable size mismatch,edge list and weights should share their first dimension"
        )

    # ###
    # edgesList = np.array([[1, 2], [3, 4], [4, 5], [5, 6]])
    # weigths = np.array([0.1, 0.2, 0.3, 0.4])
    # session_list = ()
    # ###
    # format the egdes and weigth correctly to create the graph
    weigthed_edges_list = [(v[0], v[1], w) for v, w in zip(edgesList, weigths)]
    big_G = nx.Graph()
    big_G.add_weighted_edges_from(weigthed_edges_list)
    # set a "session" attribute
    nx.set_node_attributes(big_G, session_list, "session")

    # initial cleaning of the clusters by removing dead end clusters
    # if they are duplicates
    big_G = clean_clusters(big_G)


def clean_clusters(G: nx.Graph()):
    # make list of id of connected nodes, our putative clusters
    list_clusters = [c for c in nx.connected_components(G)]

    for it_clust in list_clusters:
        # take subgraph with only nodes from one cluster
        tmp_clust = G.subgraph(it_clust)

        # now find if we have duplicate node in a layer
        tmp_sess_list = np.array([v["session"] for _, v in tmp_clust.nodes(data=True)])
        to_remove = None

        # if we have duplicate cell id in one of the session
        if len(set(tmp_sess_list)) < len(tmp_sess_list):
            # lazy use of a loop to iterate over sessions
            for id_sess in set(tmp_sess_list):
                cand_m = tmp_sess_list == id_sess
                if sum(cand_m) > 1:
                    # if the session as duplicate examine how they are:
                    # we can have 2 cases:
                    #       only one edges nodes -> keep the one with strongest weight
                    #       one multiple and other one edges -> remove all the one edge
                    id_clust = it_clust[cand_m]
                    n_edges_clust = np.array(
                        [len(tmp_clust.edges(e)) for e in id_clust]
                    )

                    if (n_edges_clust == 1).all():
                        w_edges_clust = [
                            w["weight"]
                            for _, _, w in tmp_clust.edges(id_clust, data=True)
                        ]
                        to_remove = id_clust[np.argmin(w_edges_clust)]
                        # remove dead end node with smaller weight
                        # (here we could better handle equal weight)
                        G.remove_node(to_remove)

                    elif (n_edges_clust == 1).any():
                        to_remove = id_clust[n_edges_clust == 1]

                    # remove dead end node
                    # I think we need to do this as the function to
                    # remove one or several nodes is different...
                    if to_remove is not None:
                        if to_remove.size == 1:
                            G.remove_node(int(to_remove))
                        elif to_remove.size > 1:
                            G.remove_nodes_from(to_remove)
    return G


def find_duplicated_nodes(G):

    has_duplicate = []
    no_duplicate = []

    list_clusters = [c for c in nx.connected_components(G)]

    for it_clust in list_clusters:
        # take subgraph with only nodes from one cluster
        tmp_clust = G.subgraph(it_clust)

        sess_list = np.array([v["session"] for _, v in tmp_clust.nodes(data=True)])
        node_id = np.array(tmp_clust.nodes())

        if len(set(sess_list)) < len(sess_list):

            for id_sess in set(sess_list):
                cand_m = sess_list == id_sess
                if sum(cand_m) > 1:
                    # if the session as duplicate store the id
                    has_duplicate.append(node_id[cand_m])
                else:
                    no_duplicate.append(node_id[cand_m])

    return has_duplicate, no_duplicate


def remove_duplicate(G):
    list_clusters = [c for c in nx.connected_components(G)]

    for it_clust in list_clusters:

        dup_cells, no_dup_cells = find_duplicated_nodes(G.subgraph(it_clust))
        all_combination = list(product(*dup_cells))
        all_weigth = [None] * len(all_combination)
        all_node_out = [None] * len(all_combination)

        for i, e in enumerate(all_combination):
            # look at the weigth for a subset of nodes
            tmp_nodes = no_dup_cells + list(e)
            tmp_graph = G.subgraph(tmp_nodes)

            # iterate over all the possible combinations, split into subgraph
            # in case we create one by removing a node
            # keep their id and the weigth
            all_node_out[i] = [e for e in nx.connected_components(tmp_graph)]
            all_weigth[i] = [
                G.subgraph(e).size(weight="weight")
                for e in nx.connected_components(tmp_graph)
            ]

            w = [sum(a) for a in all_weigth]
            best_clustering = np.argmax(w)

            best_clusters = all_combination[best_clustering]
