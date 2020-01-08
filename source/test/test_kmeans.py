import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from ..utils import datagen
from ..utils import dataplot
from ..clustering import kmeans
from ..clustering import stats

sns.set()
plt.rcdefaults()

def get_data():
    N = 500
    x0 = datagen.generate_gauss_data([1, 1.5], [[2, 0.5], [0.5, 0.6]], N)
    x1 = datagen.generate_gauss_data([13, 10], [[2, 0.8], [0.8, 1]], N)
    x2 = datagen.generate_gauss_data([12, 2], [[2, -0.5], [-0.5, 0.6]], N)
    x3 = datagen.generate_gauss_data([2, 10], [[2, -2], [-2, 0.5]], N)

    # Kmeans
    data = np.concatenate((x0, x1, x2, x3), axis=0)
    return shuffle(data)

if __name__ == '__main__':
    # Generate 4 separable classes
    data = get_data()

    # Clustering with kmeans
    assignments, centers, iters = kmeans.kmeans(data, 4, to_inform=True)

    # Stats
    print('Experimenting with Kmeans ...')
    stats.clustering_stats(get_data, 4)