import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from ..utils import datagen
from ..utils import dataplot
from ..classifiers.LinearClassifier import LinearClassifier
from ..classifiers.Classifier import Data

sns.set()
plt.rcdefaults()

def get_data(plt):
    N = 500
    M1 = [3, 3]
    S1 = [[2, 0.5], [0.5, 0.6]]
    M2 = [7.5, 8]
    S2 = [[2, -0.8], [-0.8, 1]]
    x1 = datagen.generate_gauss_data(M1, S1, N)
    x2 = datagen.generate_gauss_data(M2, S2, N)

    plt = dataplot.data_plot(plt, np.array([x1, x2]))

    X = np.concatenate((x1, x2), axis=0)
    y = np.repeat([0, 1], N)
    X, y = shuffle(X, y)
    data = Data(X, y, M1, S1, M2, S2)
    return plt, data

def plot_data_classifier(X, figure):
    figure.title('Data')
    figure.show()

if __name__ == '__main__':
    # Generate and plot data
    figure_data, data = get_data(plt)

    # Fit with Linear classsifier
    lin = LinearClassifier()
    lin.fit(data)

    # Predict and find error for the fitted parameters
    e = lin.prediction_error(data['X'], data['y'])
    print('Linear Classifier Error: {}/{}'.format(e, len(data['y'])))

    # Plot prediction line
    x, y = lin.get_prediction_line(data['X'])
    figure_data.plot(x, y, color='black')
    figure_data.show()
