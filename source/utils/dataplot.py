import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import ticker, cm

sns.set()
plt.rcdefaults()
data_colors = ['bo', 'ro', 'go', 'yo']

def data_plot(plt, x):
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
    f_max = Z.max()

    # Plot pdf
    f = plt.contourf(X, Y, Z, cmap=cmap)
    plt.colorbar(f)
    plt.title(title)
    plt.show()