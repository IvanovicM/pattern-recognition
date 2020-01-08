import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from . import kmeans
from . import squarerr

sns.set()
plt.rcdefaults()

def clustering_stats(get_data, k, method_iters=100, method='Kmeans'):
    iters = np.zeros(method_iters)

    for method_iter in range(method_iters):
        data = get_data()
        if method == 'Kmeans':
            assignments, centers, it = kmeans.kmeans(data, k)
        else:
            assignments, it = squarerr.square_error(data, k)
        
        iters[method_iter] = it

    # Print average number of iterations
    avg_iter = np.mean(iters)
    print('Experiment ran {} time.s\n'
          'Average number of iterations is {}.'.format(method_iters, avg_iter))

    # Plot stats
    plt.plot(iters, label='iterations')
    plt.plot(np.repeat(avg_iter, method_iters), label='average')
    plt.legend()
    plt.title('{} experiments'.format(method))
    plt.show()
