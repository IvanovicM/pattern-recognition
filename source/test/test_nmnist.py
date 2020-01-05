import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ..nmnist import dataset
from ..utils import dataplot

sns.set()
plt.rcdefaults()

if __name__ == '__main__':
    X = dataset.read_data()