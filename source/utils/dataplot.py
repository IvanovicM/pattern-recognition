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

def plot_bimodal_gauss(plt, f, x1=-10, x2=10, x3=-10, x4=10):
    x = np.linspace(x1, x2, 50)
    y = np.linspace(x3, x4, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # Calculate f(x, y) in the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(X[i, j], Y[i, j])
    f_max = Z.max()

    # Plot pdf
    levels = [0.3*f_max, 0.6*f_max, 0.8*f_max]
    f = plt.contourf(X, Y, Z)
    plt.colorbar(f)
    plt.title('Probability density function')
    plt.show()