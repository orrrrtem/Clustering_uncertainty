import numpy as np
from scipy import stats
from . import parallel


def fechner_corr(x,y):
    x_div = x -np.mean(x)
    y_div = y - np.mean(y)
    return np.sum(np.sign(x_div * y_div)) / x.shape[0], 0

# Student's T random variable
def multivariate_t_rvs(m, S, n=1, df=3):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = np.random.chisquare(df, n) / df
    z = np.random.multivariate_normal(np.zeros(d), S, (n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

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

def generate_samples_bag(mean, cov, bags = 10, sample_size = 50, distribution = np.random.multivariate_normal, **kwargs):
    if distribution.__name__ == 'multivariate_t_rvs' and len(kwargs):
        return np.hsplit(distribution(mean, cov, sample_size * bags, kwargs['df']).T, bags)
    else:
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

class corr_estimate_parallel(object):
    def __init__(self, samples_bags, corr_estimator, backend = 'threading'):
        self.samples_bags = samples_bags
        self.corr_estimator = corr_estimator
        self.backend = backend
    
    def get_estimations(self):
        indexes = [i for i in range(len(self.samples_bags))]
        estimations = parallel.For(indexes, backend  = self.backend, progress=False)(self.get_estimate_bag)
        return estimations
    def get_estimate_bag(self, index):
        indexes = [(index, i) for i in range(len(self.samples_bags[index]))]
        return parallel.For(indexes, backend  = self.backend, progress=False)(self._estimate_sample)

    def _estimate_sample(self, index):
        return set_zero_weights_to_very_low(get_corr_estimate(self.samples_bags[index[0]][index[1]], self.corr_estimator))


# def get_true_graph():
#     mean_covs = [get_mean_cov(num_clusters = num_clusters, cluster_size = cluster_size, r_in = rs[0][i], r_out = rs[1][i]) for i in range(rs.shape[1])]
#     means = [mean_cov[0] for mean_cov in mean_covs]
#     covs = [mean_cov[1] for mean_cov in mean_covs]
#     true_graph = get_cor_from_cov(cov)
#     return true_graph