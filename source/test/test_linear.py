import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from ..utils import datagen
from ..utils import dataplot
from ..classifiers import LinearClassifier

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
    data = LinearClassifier.Data(X, y, M1, S1, M2, S2)
    return plt, data

if __name__ == '__main__':
    # Generate and plot data
    plt_data, data = get_data(plt)
    #plt_data.title('Data')
    #plt_data.show()

    # Fit with Linear classsifier
    lin = LinearClassifier.LinearClassifier()
    lin.fit(data)

    # Predict and find error for the fitted parameters
    e = lin.prediction_error(data['X'], data['y'])
    print('error: {}'.format(e))
