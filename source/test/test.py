import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ..utils import datagen

sns.set()
plt.rcdefaults()

if __name__ == '__main__':
    N = 500

    x1 = datagen.generate_gauss_data([3, 3], [[2, 0.5], [0.5, 0.6]], N)
    x2 = datagen.generate_gauss_data([7.5, 8], [[2, 0.8], [1, 0.8]], N)

    x3 = datagen.generate_uniform_doughnut_part(5.5, 7.5, 0.25, 1.5, N)
    x4 = datagen.generate_uniform_circle(6, 6.5, N)

    datagen.plot_data(np.array([x1, x2, x3, x4]))





