import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set()
plt.rcdefaults()

N = 500

M1 = [3, 3]
S1 = [[2, 0.9], [0.5, 0.6]]
x1 = np.random.multivariate_normal(M1, S1, N)

M2 = [7.5, 8]
S2 = [[2, 0.8], [1, 0.8]]
x2 = np.random.multivariate_normal(M2, S2, N)

plt.plot(x1[:, 0], x1[:, 1], 'ro', x2[:, 0], x2[:, 1], 'bs')
plt.show()





