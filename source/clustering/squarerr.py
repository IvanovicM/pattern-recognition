import numpy as np
import random
from ..utils import dataplot

def get_cluster_h(X, M, S, P):
    # TODO MATRICES
    return 0.5

def square_error(data, k):
    # Initial values
    n_examples = data.shape[0]
    assignments = [0] * n_examples
    #dataplot.clusters_plot(data[:, 0], data[:, 1], assignments, title='Data')
    
    # Stop criterions
    changed = n_examples
    iters = 0
    max_iters = 100
    
    # All iterations
    while changed > 0 and iters < max_iters:
        P = np.zeros(k)
        M = np.zeros((k, 2))
        S = np.zeros((k, 2, 2))

        # Estimation for P, mean and cov
        for i in range(k):
            cluster_idx = assignments[i]
            P[cluster_idx] += 1

            all_in_cluster = np.matrix([
                data[i, :]
                for i in range(n_examples)
                if assignments[i] == cluster_idx
            ])

            M[i, :] = np.mean(all_in_cluster, axis=0)
            S[i, :, :] = np.cov(np.transpose(all_in_cluster))
        for i in range(k):
            P[i] /= n_examples

        # Remember current h
        current_h = np.zeros(n_examples)
        changed = 0
        for i in range(n_examples):
            cluster_idx = assignments[i]
            current_h[i] = get_cluster_h(
                data[i, :],
                M[cluster_idx, :], S[cluster_idx, :, :], P[cluster_idx]
            )

        # Calculate h for every other class and update assignments if needed
        for i in range(n_examples):
            is_changed = False
            
            for j in range(k):
                h = get_cluster_h( data[i, :], M[j, :], S[j, :, :], P[j])
                if h < current_h[i]:
                    is_changed = True
                    current_h[i] = h 
                    assignments[i] = j
            
            if is_changed:
                changed += 1
        
        # Output for the current iteration
        print('Iteration {}: changed examples = {}'.format(iters, changed))

        iters += 1 

    # dataplot.clusters_plot(data[:, 0], data[:, 1], assignments,
    #                        title='Final clusters')
    return assignments, iters