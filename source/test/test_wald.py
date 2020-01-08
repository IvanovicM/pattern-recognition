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

def get_x1():
    N = 500

    # Class 1
    M1 = [1, 1]
    S1 = [[4, 1.1], [1.1, 2]]
    M2 = [6, 4]
    S2 = [[3, -0.8], [-0.8, 1.5]]
    P1 = 0.6
    return datagen.generate_bimodal_gauss(M1, S1, M2, S2, P1, N)

def get_x2():
    N = 500

    # Class 2
    M1 = [1, 2]
    S1 = [[2, 1.1], [1.1, 4]]
    M2 = [6, 3]
    S2 = [[3, 0.8], [0.8, 0.5]]
    P1 = 0.55
    return datagen.generate_bimodal_gauss(M1, S1, M2, S2, P1, N)

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

def f1_vec(x):
    return f1(x[0], x[1])

def f2_vec(x):
    return f2(x[0], x[1])

def plot_m_of_eps(wald, fixed_eps, eps_change=1):
    # Placeholders for m-s and eps-s
    all_eps_pow = np.arange(-10, 0, 0.1)
    all_eps = np.array([pow(10, pw) for pw in all_eps_pow])
    all_m = np.zeros(all_eps.shape)

    for i in range(len(all_eps)):
        # Fix epsilons
        if eps_change == 1:
            eps1 = all_eps[i]
            eps2 = fixed_eps
        else:
            eps1 = fixed_eps
            eps2 = all_eps[i]

        # Remember results
        y1 = wald.predict_class(get_x1(), eps1, eps2)
        all_m[i] = wald['m']
        y2 = wald.predict_class(get_x2(), eps1, eps2)
        all_m[i] += wald['m']

        all_m[i] /= 2

    # Plot graphic
    plt.semilogx(all_eps, all_m)
    plt.xlabel('Eps{}'.format(eps_change))
    plt.ylabel('m')
    plt.title('Eps{} = {}'.format(abs(2 - eps_change) + 1, fixed_eps))
    plt.show()

if __name__ == '__main__':
    # Generate and plot data
    x1 = get_x1()
    x2 = get_x2()

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
    plot_m_of_eps(wald, 1e-5, eps_change=1)
    plot_m_of_eps(wald, 1e-5, eps_change=2)
