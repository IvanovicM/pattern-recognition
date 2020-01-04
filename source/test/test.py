import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ..utils import datagen

sns.set()
plt.rcdefaults()

if __name__ == '__main__':
    N = 500

    x1 = datagen.generate_gauss_data([3, 3], [[2, 0.9], [0.5, 0.6]], N)
    x2 = datagen.generate_gauss_data([7.5, 8], [[2, 0.8], [1, 0.8]], N)

    datagen.plot_data_two_classes(x1, x2)





