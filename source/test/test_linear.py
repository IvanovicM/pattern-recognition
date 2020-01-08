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

if __name__ == '__main__':
    # Generate and plot data
    figure_data, data = get_data(plt)

    # Fit with Linear classsifier
    lin = LinearClassifier()

    lin.fit(data, method='resubstitution')
    e_r = lin.prediction_error(data['X'], data['y'])

    lin.fit(data, method='desired_output')
    e_do = lin.prediction_error(data['X'], data['y'])

    # Print errors
    print('Linear Classifier (Resubstitution Approach) Error: {}/{}'.format(
          e_r, len(data['y']))
    )
    print('Linear Classifier (Desired Output Approach) Error: {}/{}'.format(
          e_do, len(data['y']))
    )

    # Spatial results
    def f(x, y):
        return lin.predict_classes(np.array([x, y]))
    figure_data.legend()
    dataplot.plot_f(figure_data, f, -2, 12, 0, 12, cmap='binary',
                    title='Divided space')