import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ..utils import datagen
from ..utils import dataplot
from ..kmeans import kmeans

sns.set()
plt.rcdefaults()

if __name__ == '__main__':
    # Generate 4 classes
    N = 500
    x0 = datagen.generate_gauss_data([1, 1.5], [[2, 0.5], [0.5, 0.6]], N)
    x1 = datagen.generate_gauss_data([13, 10], [[2, 0.8], [0.8, 1]], N)
    x2 = datagen.generate_gauss_data([12, 2], [[2, -0.5], [-0.5, 0.6]], N)
    x3 = datagen.generate_gauss_data([2, 10], [[2, -2], [-2, 0.5]], N)

    #plt = dataplot.data_plot(plt, np.array([x0, x1, x2, x3]))
    #plt.title('Classes')
    #plt.show()

    data = np.concatenate((x0, x1, x2, x3), axis=0)
    # assignments, centers = kmeans.kmeans(data, 4)