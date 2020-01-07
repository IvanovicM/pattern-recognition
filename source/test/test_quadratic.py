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

    # Class 0
    c_x1 = 5.5
    c_x2 = 7.5
    angle = 0.25
    dist = 1.5
    x1 = datagen.generate_uniform_doughnut_part(c_x1, c_x2, angle, dist, N)

    # Class 1
    c_x1 = 6
    c_x2 = 6.5
    x2 = datagen.generate_uniform_circle(c_x1, c_x2, N)

    plt = dataplot.data_plot(plt, np.array([x1, x2]))

    X = np.concatenate((x1, x2), axis=0)
    y = np.repeat([0, 1], N)
    X, y = shuffle(X, y)
    data = Data(X, y, np.mean(x1, axis=1), None, np.mean(x1, axis=1), None)
    return plt, data

if __name__ == '__main__':
    # Generate and plot data
    figure_data, data = get_data(plt)

    # Fit with Linear classsifier
    quadr = QuadraticClassifier()

    quadr.fit(data, method='desired_output')
    e_do = quadr.prediction_error(data['X'], data['y'])
    x_do, y_do = quadr.get_prediction_line(data['X'], eps=0.01)

    print('Quadratic Classifier (Desired Output Approach) Error: {}/{}'.format(
          e_do, len(data['y']))
    )

    # Plot Results
    figure_data.plot(x_do, y_do, color='black')
    figure_data.title('Classification')
    figure_data.legend()
    figure_data.show()