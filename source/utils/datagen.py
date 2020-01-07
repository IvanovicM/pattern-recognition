import numpy as np
import math
from sklearn.utils import shuffle

def generate_gauss_data(M, S, N):
    return np.random.multivariate_normal(M, S, N)

def generate_uniform_circle(center_x1, center_x2, N):
    x = np.zeros((N, 2))
    for i in range(N):
        alpha = 2 * math.pi * np.random.uniform()
        R = 2 * np.random.uniform()

        x1 = center_x1 + R * math.cos(alpha)
        x2 = center_x2 + R * math.sin(alpha)

        x[i, :] = [x1, x2]
    return x

def generate_bimodal_gauss(M1, S1, M2, S2, P1, N):
    N1 = int(P1 * N)
    N2 = N - N1

    x1 = generate_gauss_data(M1, S1, N1)
    x2 = generate_gauss_data(M2, S2, N2)
    X = np.concatenate((x1, x2), axis=0)
    return shuffle(X)

def generate_uniform_doughnut_part(center_x1, center_x2, angle, distance, N):
    x = np.zeros((N, 2))
    for i in range(N):
        alpha = ((1 - angle) * np.random.uniform() + angle) * math.pi
        R = np.random.uniform() + distance

        x1 = center_x1 + R * math.cos(alpha)
        x2 = center_x2 + R * math.sin(alpha)

        x[i, :] = [x1, x2]
    return x