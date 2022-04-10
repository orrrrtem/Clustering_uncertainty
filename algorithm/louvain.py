import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities

def louvain(adj_matrix, num_clusters = 0):
    graph = nx.from_numpy_array(adj_matrix)
    comms = louvain_communities(graph)
    labels_ = np.zeros(graph.number_of_nodes(), dtype=int)
    for k, comm in enumerate(comms):
        for vertex in comm:
            labels_[vertex] = k
    return  labels_