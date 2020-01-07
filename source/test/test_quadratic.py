import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from ..utils import datagen
from ..utils import dataplot
from ..classifiers.QuadraticClassifier import QuadraticClassifier
from ..classifiers.Classifier import Data

sns.set()
plt.rcdefaults()

def get_data(plt):
    N = 500
    M1 = [3, 3]
    S1 = [[2, 0.5], [0.5, 0.6]]
    M2 = [7.5, 8]
    S2 = [[2, -0.8], [-0.8, 1]]
    x1 = datagen.generate_uniform_doughnut_part(5.5, 7.5, 0.25, 1.5, N)
    x2 = datagen.generate_uniform_circle(6, 6.5, N)

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
    lin = QuadraticClassifier()

    lin.fit(data, method='resubstitution')
    e_r = lin.prediction_error(data['X'], data['y'])
    x_r, y_r = lin.get_prediction_line(data['X'], eps=0.1)

    lin.fit(data, method='desired_output')
    e_do = lin.prediction_error(data['X'], data['y'])
    x_do, y_do = lin.get_prediction_line(data['X'], eps=0.01)

    # Print errors
    print('Linear Classifier (Resubstitution Approach) Error: {}/{}'.format(
          e_r, len(data['y']))
    )
    print('Linear Classifier (Desired Output Approach) Error: {}/{}'.format(
          e_do, len(data['y']))
    )

    # Plot Results
    figure_data.plot(x_r, y_r, color='purple', label='Resubstitution Approach')
    figure_data.plot(x_do, y_do, color='black', label='Desired Output Approach')
    figure_data.title('Classification')
    figure_data.legend()
    figure_data.show()