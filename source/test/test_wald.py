import seaborn as sns
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import multivariate_normal
from ..utils import datagen
from ..utils import dataplot
from ..classifiers.WaldTest import WaldTest

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
    M1 = [1, 2]
    S1 = [[2, 1.1], [1.1, 4]]
    M2 = [6, 3]
    S2 = [[3, 0.8], [0.8, 0.5]]
    P1 = 0.55
    x2 = datagen.generate_bimodal_gauss(M1, S1, M2, S2, P1, N)

    plt = dataplot.data_plot(plt, np.array([x1, x2]))
    return plt, x1, x2

def f1(x, y):
    M1 = [1, 1]
    S1 = [[4, 1.1], [1.1, 2]]
    M2 = [6, 4]
    S2 = [[3, -0.8], [-0.8, 1.5]]
    P1 = 0.6

    return P1 * multivariate_normal(mean=M1, cov=S1).pdf(np.array([x, y])) + (
           (1-P1) * multivariate_normal(mean=M2, cov=S2).pdf(np.array([x, y])))

def f2(x, y):
    M1 = [1, 2]
    S1 = [[2, 1.1], [1.1, 4]]
    M2 = [6, 3]
    S2 = [[3, 0.8], [0.8, 0.5]]
    P1 = 0.55

    return P1 * multivariate_normal(mean=M1, cov=S1).pdf(np.array([x, y])) + (
           (1-P1) * multivariate_normal(mean=M2, cov=S2).pdf(np.array([x, y])))

def f(x, y):
    return f1(x, y) + f2(x, y)

def f1_vec(x):
    return f1(x[0], x[1])

def f2_vec(x):
    return f2(x[0], x[1])

def plot_m_of_eps(wald, x, eps, class_num_ch=0, real_class=0):
    if class_num_ch == 0:
        eps2 = eps
    else:
        eps1 = eps

    # Placeholders for eps-s and m-s
    all_eps_pow = np.arange(-10, 0, 0.2)
    all_eps = np.array([pow(10, pw) for pw in all_eps_pow])
    all_m = np.zeros(all_eps.shape)

    for i in range(len(all_eps)):
        # Fix epsilons
        if class_num_ch == 0:
            eps1 = all_eps[i]
        else:
            eps2 = all_eps[i]

        # Remebers results
        y = wald.predict_class(x, eps1, eps2)
        all_m[i] = wald['m']

    # Plot graphic
    plt.plot(all_eps, all_m)
    plt.title('Eps{} = {}'.format(abs(1-class_num_ch), eps))
    plt.show()

if __name__ == '__main__':
    # Generate and plot data
    figure_data, x1, x2 = get_data(plt)
    figure_data.show()

    # Wald Test
    wald = WaldTest(f1_vec, f2_vec)

    # Predict for the first class
    y1 = wald.predict_class(x1, 1e-4, 1e-5)
    print('Predicted class: {}. Real class: 0. Steps number: {}.'.format(
          y1, wald['m']))

    # Predict for the second class
    y2 = wald.predict_class(x2, 1e-4, 1e-5)
    print('Predicted class: {}. Real class: 1. Steps number: {}.'.format(
          y2, wald['m']))

    # Try various epsilons and plot results
    plot_m_of_eps(wald, x1, 1e-5, class_num_ch=0)
    plot_m_of_eps(wald, x2, 1e-5, class_num_ch=0)
    plot_m_of_eps(wald, x1, 1e-5, class_num_ch=1)
    plot_m_of_eps(wald, x2, 1e-5, class_num_ch=1)
