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
from ..classifiers.LinearClassifier import LinearClassifier

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

def f(x, y):
    return f1(x, y) + f2(x, y)

def f1_vec(x):
    return f1(x[0], x[1])

def f2_vec(x):
    return f2(x[0], x[1])

if __name__ == '__main__':
    # Plot bimodal pdf for the data
    dataplot.plot_f(plt, f1, -3, 10, -2.5, 7.5,
                      title='Probability density function')
    dataplot.plot_f(plt, f2, 2.5, 10, -12.5, -2.5,
                      title='Probability density function')
    dataplot.plot_f(plt, f, -3, 10, -12.5, 7.5,
                      title='Probability density functions')

    # Generate and plot data
    figure_data, data = get_data(plt)

    # Predict outputs with Bayes classsifier
    bayes = BayesClassifier(f1_vec, f2_vec)
    Y = bayes.predict_classes(data['X'])

    # Print errors
    e = bayes.prediction_error(data['X'], data['y'])
    print('Bayes Classifier Error: {}/{}'.format(e, len(data['y'])))

    # Spatial results
    def f_bayes(x, y):
        return bayes.predict_classes(np.array([x, y]))
    figure_data.legend()
    dataplot.plot_f(figure_data, f_bayes, -7, 14, -17, 10, cmap='binary',
                    title='Divided space - Bayes')

    # Compare with linear classifier
    lin = LinearClassifier()
    lin.fit(data, method='desired_output')

    e_lin = bayes.prediction_error(data['X'], data['y'])
    print('Linear Classifier (Desired Output Approach) Error: {}/{}'.format(
          e_lin, len(data['y']))
    )

    def f_lin(x, y):
        return lin.predict_classes(np.array([x, y]))
    figure_data.legend()
    dataplot.plot_f(figure_data, f_lin, -7, 14, -17, 10, cmap='binary',
                    title='Divided space - Linear')
