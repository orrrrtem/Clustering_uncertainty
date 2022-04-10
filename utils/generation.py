import numpy as np
from scipy import stats

def get_mean_cov(num_clusters = 2, cluster_size = 50, r_in = 1, r_out = 0):
    vertex_count = num_clusters * cluster_size
    mean = np.zeros(vertex_count)
    r_ins = np.full((cluster_size, cluster_size), r_in)
    r_outs = np.full((cluster_size, cluster_size), r_out)
    cov = np.block([[np.tile(r_outs,k),r_ins,np.tile(r_outs,num_clusters-k -1)]  for k in range(num_clusters)])
    np.fill_diagonal(cov,1)
    return mean, cov

def get_cor_from_cov(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def generate_samples_bag(mean, cov, bags = 10, sample_size = 50, distribution = np.random.multivariate_normal):
    return np.hsplit(distribution(mean, cov, sample_size * bags).T, bags)

#this function is required since networkx does not link vertices with zero weigth
#TODO - remove this solutiuion as it slightly moves distribution if zero is in sample
def set_zero_weights_to_very_low(adj_matrixes, value = 1e-6):
    adj_matrixes[adj_matrixes < value] = value
    return adj_matrixes

def get_corr_estimate(sample, corr_estimator = stats.pearsonr):
    vertex_count = sample.shape[0]
    corr_estimate = np.ones((vertex_count, vertex_count))
    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            corr, _ = corr_estimator(sample[i], sample[j])
            corr = abs(corr)
            corr_estimate[i][j] = corr
            corr_estimate[j][i] = corr
    
    return corr_estimate


