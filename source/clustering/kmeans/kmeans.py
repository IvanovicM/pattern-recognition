import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

sns.set()
plt.rcdefaults()
cmap = ListedColormap(['red', 'green', 'blue', 'magenta'])

def clusters_plot(X, Y, assignments=None, centers=None, title=None):
    if assignments is None:
        assignments = [0] * len(X)
    fig = plt.figure(figsize=(14,8))    
    
    # Plot colored data and clusters center
    plt.scatter(X, Y, c=assignments, cmap=cmap)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1],
                    marker='+', s=400, color='black', label='cluster center')
    plt.title(title)
    plt.legend()
    plt.show()

def kmeans(data, k):
    # Initial values
    n_examples = data.shape[0]
    centers = data[random.sample(range(n_examples), k)]
    assignments = [0] * n_examples
    clusters_plot(data[:, 0], data[:, 1], assignments, centers,
                 title='Data and initial centers')
    
    # Stop criterions
    delta = 100
    eps = 1e-3
    iters = 0
    max_iters = 100
    previous_centers = centers
    
    # All iterations
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

    clusters_plot(data[:, 0], data[:, 1], assignments, centers,
                  title='Final clusters and their centers')
    return assignments, centers, iters