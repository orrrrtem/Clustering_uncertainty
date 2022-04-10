import sklearn
import numpy as np
import networkx as nx
import numpy.linalg as la
import scipy.cluster.vq as vq
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import csv
import warnings
warnings.filterwarnings('ignore')

from tqdm.auto import tqdm

from scipy.sparse import csgraph
from sklearn.metrics.cluster import rand_score
#from sklearn.metrics import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score

import networkx.algorithms.community as nx_comm

from .generation import get_mean_cov, get_cor_from_cov, generate_samples_bag, set_zero_weights_to_very_low, get_corr_estimate

#from matplotlib.pyplot import figure


def get_true_labels(num_groups, num_members):
    true_labels = np.zeros(num_groups * num_members, dtype=int)
    for i in range(0, num_groups):
        for j in  range(0, num_members):
            true_labels[i*num_members + j] = i
    return true_labels

def get_partition(labels):
    partition = []
    for i in range(labels.max() + 1):
        partition.append(np.where(labels == i)[0])
    return partition


def get_rs_by_relation(r_out = 0.1, relation = np.linspace( 1, 10, 10)):
    r_outs = np.full(relation.shape[0], r_out)
    r_ins = np.array([x*r_out for x in relation])
    return np.vstack((r_ins,r_outs))

def get_rs_from_fixed_rin(r_in = 0.8, count_rout = 20, epsilon = 0):
    r_ins = np.full(count_rout, r_in)
    r_outs = np.linspace( 0, r_in + epsilon, count_rout)
    return np.vstack((r_ins,r_outs))

def get_rs_from_fixed_weighted_degree(degree=16, cluster_size= 20, num_clusters=2,r_out_bound = (0,1,20)):
    #D = (N-1)D1 + (K-1)nD2
    #D1= (D - (K-1)ND2)/(N-1)
    #D2MAX > D/(KN - 1)

    r_outs = np.linspace( r_out_bound[0], r_out_bound[1], r_out_bound[2])
    r_outs = r_outs[r_outs <=degree/(num_clusters*cluster_size - 1)]
    r_ins = (degree - (num_clusters - 1)*cluster_size*r_outs)/(cluster_size - 1)
    return np.vstack((r_ins,r_outs))
    

def compute_clustering(rs, algos, num_clusters = 2, cluster_size=5, sample_vol = 10, num_repeats = 200):
    mean_covs = [get_mean_cov(num_clusters = num_clusters, cluster_size = cluster_size, r_in = rs[0][i], r_out = rs[1][i]) for i in range(rs.shape[1])]
    means = [mean_cov[0] for mean_cov in mean_covs]
    covs = [mean_cov[1] for mean_cov in mean_covs]
    true_graphs = [get_cor_from_cov(cov) for cov in covs]
    [set_zero_weights_to_very_low(true_graph) for true_graph in true_graphs]
    samples_bags = [generate_samples_bag(means[i], covs[i], bags = num_repeats, sample_size=sample_vol) for i,cov in enumerate(covs)]
    print('Generating graphs started')
    estimated_graphs_bags = [[set_zero_weights_to_very_low(get_corr_estimate(sample)) for sample in samples_bag] for samples_bag in tqdm(samples_bags)]
    print('Generating graphs complete')

    true_labels = get_true_labels(num_clusters, cluster_size)

    result = dict()    

    for algo in algos:
        algo_result = []
        print(algo.__name__ + ' started')
        for idx, estimated_graphs_bag in enumerate(tqdm(estimated_graphs_bags)):
            repeat_result = []
            for estimated_graph in estimated_graphs_bag:
                repeat_result.append(algo(estimated_graph, num_clusters))
            algo_result.append(repeat_result)
                
        print(algo.__name__ + ' complete')
        result[algo.__name__] = algo_result

    return true_labels, result, estimated_graphs_bags

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def nested_dict_to_dict(x):
    y = dict()
    for k1 in x:
        for k2 in x[k1]:
            y[(k1,k2)]=x[k1][k2]
    return y

def plot_quality_by_relation(rs, metrics, concrete_metrics=None, by_rin = True):
    #figure(figsize=(10, 12), dpi=100)
    fig, ax = plt.subplots()

    colors = ['-b', '-r', '-g', '-y', '-p', '-c', '-m']
    #if len(metrics) * len(list(metrics.values())[0]) > len(colors):
    reduced_metrics = nested_dict_to_dict(metrics)
    #colors = get_cmap(len(reduced_metrics))
    c = 0
    if by_rin:
        r=rs[0]
    else:
        r=rs[1]
    if(concrete_metrics):
        for i, algo_metric in enumerate(reduced_metrics):
            if algo_metric[1] is concrete_metrics:
                ax.plot(r, reduced_metrics[algo_metric], colors[c], label=algo_metric[1] + ' ' + algo_metric[0])
                c+=1
    else:
        for i, algo_metric in enumerate(reduced_metrics):
                ax.plot(r, reduced_metrics[algo_metric], colors[c], label=algo_metric[1] + ' ' + algo_metric[0])
                c+= 1
    #ax.axis('equal')#ax.axis('p_in')
    leg = ax.legend()
    #plt.plot(p_ins[:len(metric)], metric)
    plt.ylabel('Metric value', size=10)
    if by_rin:
        plt.xlabel('R_in', size=10)
        plt.title('R_out=' + str(rs[1][0]), size=14)

    else:
        plt.xlabel('R_out', size=10)
        #plt.title('R_out=' + str(rs[0][1]), size=14)
    return


def validation(rs, true_labels, result, estimated_graphs_bags, by_rin = True):
    metrics_by_algos = dict()
    for algo in tqdm(result):
        metrics = dict()
        metrics['RI'] = [np.mean(np.array([rand_score(true_labels, labels) for labels in labels_repeated])) for labels_repeated in result[algo]]
        metrics['ARI'] = [np.mean(np.array([adjusted_rand_score(true_labels, labels) for labels in labels_repeated])) for labels_repeated in result[algo]]
        metrics['MI'] = [np.mean(np.array([mutual_info_score(true_labels, labels) for labels in labels_repeated])) for labels_repeated in result[algo]]
        metrics['AMI'] = [np.mean(np.array([adjusted_mutual_info_score(true_labels, labels) for labels in labels_repeated])) for labels_repeated in result[algo]]
        #metrics['modularity'] = [np.mean(np.array([nx_comm.modularity(true_labels, get_partition(labels)) for labels in labels_repeated])) for labels_repeated in result[algo]]
        metrics_by_algos[algo] = metrics
    plot_quality_by_relation(rs, metrics_by_algos, 'RI', by_rin)
    plot_quality_by_relation(rs, metrics_by_algos, 'ARI', by_rin)
    plot_quality_by_relation(rs, metrics_by_algos, 'MI', by_rin)
    plot_quality_by_relation(rs, metrics_by_algos, 'AMI', by_rin)
    return metrics_by_algos


def metrics_to_df(metrics, rs):
    df= pd.DataFrame(nested_dict_to_dict(metrics))
    df['r_in'] = rs[0]
    df.set_index('r_in', inplace=True)
    return df
