import numpy as np
import networkx as nx
import numpy.linalg as la
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.sparse import csgraph

def spectral_clustering(adj_matrix, num_clusters = 0):
  D = np.diag(np.ravel(np.sum(adj_matrix,axis=1)))
  L = D - adj_matrix
  l, U = la.eigh(L)
  kmeans = KMeans(n_clusters=num_clusters).fit(U[:,1:num_clusters])
  labels_ = kmeans.labels_
  return  labels_

def normalized_spectral_clustering(adj_matrix, num_clusters = 0):
  l, U = la.eigh(csgraph.laplacian(adj_matrix, normed=True))
  kmeans = KMeans(n_clusters=num_clusters).fit(U[:,1:num_clusters])
  labels_ = kmeans.labels_
  return labels_