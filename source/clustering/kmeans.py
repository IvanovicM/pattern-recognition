import numpy as np
import random
from ..utils import dataplot

def kmeans(data, k, to_plot=False):
    # Initial values
    n_examples = data.shape[0]
    centers = data[random.sample(range(n_examples), k)]
    assignments = [0] * n_examples
    if to_plot:
        dataplot.clusters_plot(data[:, 0], data[:, 1], assignments, centers,
                            title='Data and initial centers')
    
    # Stop criterions
    delta = 100
    eps = 1e-3
    iters = 0
    max_iters = 100
    previous_centers = centers
    
    # All iterations
    print('Kmeans algorithm ...')
    while delta > eps and iters < max_iters:
        # Assignment step
        for i in range(n_examples):
            dists = np.linalg.norm(centers - data[i], axis=1)
            assignments[i] = np.argmin(dists)
        
        # Update step
        centers = np.zeros((k, 2))
        cluster_sizes = np.zeros(k)
        for i in range(n_examples):
            cluster_idx = assignments[i]
            centers[cluster_idx] += data[i]
            cluster_sizes[cluster_idx] += 1
        for i in range(k):
            centers[i] /= cluster_sizes[i]
        
        # Output for the current iteration
        delta = np.average(np.abs(centers - previous_centers))
        print('Iteration {}: delta = {}'.format(iters, delta))
        previous_centers = centers

        iters += 1 

    if to_plot:
        dataplot.clusters_plot(data[:, 0], data[:, 1], assignments, centers,
                            title='Final clusters and their centers')
    return assignments, centers, iters