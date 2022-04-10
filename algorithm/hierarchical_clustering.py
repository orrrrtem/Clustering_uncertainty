import networkx as nx
from networkx.algorithms import tree
import numpy as np

def get_community_labels(G):
    cc = list(nx.connected_components(G))
    labels_ = np.zeros(G.number_of_nodes(), dtype=int)
    for k, comm in enumerate(cc):
        for label in comm:
            labels_[label] = k
    return  labels_

def get_community_labels_from_less_cc(G, num_clusters):
    cc = list(nx.connected_components(G))
    labels_ = np.zeros(G.number_of_nodes(), dtype=int)

    for k, comm in enumerate(cc):
        for label in comm:
            labels_[label] = k
    return  labels_

def get_unique_thresholds(adj_matrix):
    tvs = adj_matrix.copy()
    tvs = tvs.reshape(tvs.shape[0] * tvs.shape[1])
    tvs = np.sort(tvs)
    tvs = np.unique(tvs)
    return tvs

def threshold_clustering(adj_matrix, num_clusters):
    adj_matrix_ = adj_matrix.copy()
    tvs = get_unique_thresholds(adj_matrix_)
    k = 0
    for k in range(tvs.shape[0]):
        adj_matrix_prev = adj_matrix_.copy()
        adj_matrix_[adj_matrix_ < tvs[k]] = 0
        G = nx.from_numpy_array(adj_matrix_)
        num_components = nx.number_connected_components(G)
        if num_components < num_clusters:
            continue
        elif num_components == num_clusters:
            return get_community_labels(G)
        else: 
            for i in range(adj_matrix_prev.shape[0]):
                for j in range(adj_matrix_prev.shape[0]):
                    if (adj_matrix_prev[i][j] < tvs[k] and adj_matrix_prev[i][j] != 0):
                        adj_matrix_prev[i][j] = 0
                        G = nx.from_numpy_array(adj_matrix_prev)
                        num_components = nx.number_connected_components(G)
                        if num_components == num_clusters:
                            return get_community_labels(G)
    for i in range(adj_matrix_.shape[0]):
        for j in range(adj_matrix_.shape[0]):
            if (adj_matrix_[i][j] != 0):
                adj_matrix_[i][j] = 0
                G = nx.from_numpy_array(adj_matrix_)
                num_components = nx.number_connected_components(G)
                if num_components == num_clusters:
                    return get_community_labels(G)

    return np.arange(adj_matrix.shape[0])




def mst_cut_clustering(adj_matrix, num_clusters):
    G = nx.from_numpy_array(adj_matrix)
    mst = tree.maximum_spanning_edges(G, algorithm="kruskal")
    edgelist = list(mst)
    edgelist.sort(key=lambda tup: tup[2]['weight'])
    cutted_mst = nx.from_edgelist(edgelist) 
    to_cut = edgelist[0:num_clusters - 1]
    cutted_mst.remove_edges_from(to_cut)
    return get_community_labels(cutted_mst)