import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set()
plt.rcdefaults()

def generate_gauss_data(M, S, N):
    return np.random.multivariate_normal(M, S, N)

def plot_data_two_classes(x1, x2):
    plt.plot(x1[:, 0], x1[:, 1], 'ro', x2[:, 0], x2[:, 1], 'bo')
    plt.legend(['Class 1', 'Class 2'])
    plt.show()






