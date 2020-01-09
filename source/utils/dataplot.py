import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import ticker, cm
from matplotlib.colors import ListedColormap

sns.set()
plt.rcdefaults()

def data_plot(plt, x):
    data_colors = ['bo', 'ro', 'go', 'yo']
    for i in range(x.shape[0]):
        plt.plot(x[i, :, 0], x[i, :, 1], data_colors[i],
                 label='Class {}'.format(i)
        ) 

    return plt

def plot_f(plt, f, x1=-10, x2=10, x3=-10, x4=10, cmap=None, title=None):
    x = np.linspace(x1, x2, 50)
    y = np.linspace(x3, x4, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # Calculate f(x, y) in the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            value = f(X[i, j], Y[i, j])
            if isinstance(value, list):
                value = value[0]
            Z[i, j] = value

    # Plot pdf
    f = plt.contourf(X, Y, Z, cmap=cmap)
    plt.colorbar(f)
    plt.title(title)
    plt.show()

def plot_f_peaks(plt, f, x1=-10, x2=10, x3=-10, x4=10, cmap=None):
    x = np.linspace(x1, x2, 50)
    y = np.linspace(x3, x4, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # Calculate f(x, y) in the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            value = f(X[i, j], Y[i, j])
            if isinstance(value, list):
                value = value[0]
            Z[i, j] = value
    Z_max = Z.max()

    # Plot pdf
    limit = np.exp(-1/2)
    plt.contour(X, Y, Z, levels=[limit*Z_max, (limit + 0.05)*Z_max], cmap=cmap)
    limit = np.exp(-4/2)
    plt.contour(X, Y, Z, levels=[limit*Z_max, (limit + 0.05)*Z_max], cmap=cmap)
    limit = np.exp(-9/2)
    plt.contour(X, Y, Z, levels=[limit*Z_max, (limit + 0.05)*Z_max], cmap=cmap)
    return plt

def clusters_plot(X, Y, assignments=None, centers=None, title=None):
    cmap = ListedColormap(['red', 'green', 'magenta', 'cyan', 'blue'])
    if assignments is None:
        assignments = [0] * len(X)
    
    # Plot colored data and clusters center
    plt.scatter(X, Y, c=assignments, cmap=cmap)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1],
                    marker='+', s=400, color='black', label='cluster center')
        plt.legend()
    plt.title(title)
    plt.show()