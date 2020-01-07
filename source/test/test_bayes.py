import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from ..utils import datagen
from ..utils import dataplot
from ..classifiers.Classifier import Data

sns.set()
plt.rcdefaults()

def get_data(plt):
    N = 500

    # Class 1
    M1 = [1, 1]
    S1 = [[4, 1.1], [1.1, 2]]
    M2 = [6, 4]
    S2 = [[3, -0.8], [-0.8, 1.5]]
    P1 = 0.6
    x1 = datagen.generate_bimodal_gauss(M1, S1, M2, S2, P1, N)

    # Class 2
    M1 = [7, -9]
    S1 = [[2, 1.1], [1.1, 4]]
    M2 = [6, -5]
    S2 = [[3, 0.8], [0.8, 0.5]]
    P1 = 0.55
    x2 = datagen.generate_bimodal_gauss(M1, S1, M2, S2, P1, N)

    plt = dataplot.data_plot(plt, np.array([x1, x2]))

    X = np.concatenate((x1, x2), axis=0)
    y = np.repeat([0, 1], N)
    X, y = shuffle(X, y)
    data = Data(X, y, M1, S1, M2, S2)
    return plt, data

#def f(x, y):

if __name__ == '__main__':
    figure_data, data = get_data(plt)
    figure_data.show()

    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)

    # X, Y = np.meshgrid(x, y)
    # Z = f(X, Y)
    # plt.contour(X, Y, Z)
