import numpy as np
import random
from ..utils import dataplot

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def get_cluster_h(X, M, S, P):
    if not is_invertible(S):
        return 1e10
    det = np.linalg.det(S)
    d = np.subtract(X, M)
    norm = np.matmul(
            np.matmul(
                np.transpose(d), np.linalg.inv(S)
            ), d 
    )
    return 0.5 * (norm + np.log(det) - np.log(P))

def square_error(data, k, to_inform=False):
    # Initial values
    n_examples = data.shape[0]
    assignments = np.floor(k * np.random.uniform(size=n_examples))
    assignments = [int(a) for a in assignments]
    if to_inform:
        dataplot.clusters_plot(data[:, 0], data[:, 1], assignments,
                            title='Initial clusters')
    
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
        for cluster_idx in range(k):
            all_in_cluster = np.matrix([
                data[i, :]
                for i in range(n_examples)
                if assignments[i] == cluster_idx
            ])
            if all_in_cluster.shape[1] == 0:
                continue

            M[cluster_idx, :] = np.mean(all_in_cluster, axis=0)
            S[cluster_idx, :, :] = np.cov(np.transpose(all_in_cluster))
            P[cluster_idx] = (
                len([i for i in assignments if i == cluster_idx]) / n_examples
            )

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
                h = get_cluster_h(data[i, :], M[j, :], S[j, :, :], P[j])
                if h < current_h[i]:
                    is_changed = True
                    current_h[i] = h 
                    assignments[i] = j
            
            if is_changed:
                changed += 1
        
        # Output for the current iteration
        if to_inform:
            print('Iteration {}: changed examples = {}'.format(iters, changed))

        iters += 1 

    if to_inform:
        dataplot.clusters_plot(data[:, 0], data[:, 1], assignments,
                            title='Final clusters')
    return assignments, iters