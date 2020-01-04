import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

sns.set()
plt.rcdefaults()
colors = ['bo', 'ro', 'go', 'yo']

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

def generate_uniform_doughnut_part(center_x1, center_x2, angle, distance, N):
    x = np.zeros((N, 2))
    for i in range(N):
        alpha = ((1 - angle) * np.random.uniform() + angle) * math.pi
        R = np.random.uniform() + distance

        x1 = center_x1 + R * math.cos(alpha)
        x2 = center_x2 + R * math.sin(alpha)

        x[i, :] = [x1, x2]
    return x

def plot_data(x):
    legend = [''] * x.shape[0]
    for i in range(x.shape[0]):
        plt.plot(x[i, :, 0], x[i, :, 1], colors[i])
        legend[i] = 'Class {}'.format(i) 

    plt.legend(legend)
    plt.show()






