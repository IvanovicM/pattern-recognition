import seaborn as sns
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import multivariate_normal
from ..utils import datagen
from ..utils import dataplot
from ..classifiers.Classifier import Data
from ..classifiers.BayesClassifier import BayesClassifier

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

def f1(x, y):
    M1 = [1, 1]
    S1 = [[4, 1.1], [1.1, 2]]
    M2 = [6, 4]
    S2 = [[3, -0.8], [-0.8, 1.5]]
    P1 = 0.6

    return P1 * multivariate_normal(mean=M1, cov=S1).pdf(np.array([x, y])) + (
           (1-P1) * multivariate_normal(mean=M2, cov=S2).pdf(np.array([x, y])))

def f2(x, y):
    M1 = [7, -9]
    S1 = [[2, 1.1], [1.1, 4]]
    M2 = [6, -5]
    S2 = [[3, 0.8], [0.8, 0.5]]
    P1 = 0.55

    return P1 * multivariate_normal(mean=M1, cov=S1).pdf(np.array([x, y])) + (
           (1-P1) * multivariate_normal(mean=M2, cov=S2).pdf(np.array([x, y])))

if __name__ == '__main__':
    # Plot bimodal pdf for the data
    dataplot.plot_pdf(plt, f1, -3, 10, -2.5, 7.5)
    dataplot.plot_pdf(plt, f2, 2.5, 10, -12.5, -2.5)

    # Generate and plot data
    figure_data, data = get_data(plt)

    # Fit with Bayes classsifier
    bayes = BayesClassifier()
