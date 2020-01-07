import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

sns.set()
plt.rcdefaults()
data_colors = ['bo', 'ro', 'go', 'yo']

def data_plot(plt, x):
    for i in range(x.shape[0]):
        plt.plot(x[i, :, 0], x[i, :, 1], data_colors[i],
                 label='Class {}'.format(i)
        ) 

    return plt