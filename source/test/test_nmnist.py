import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from ..nmnist import dataset
from ..utils import dataplot

sns.set()
plt.rcdefaults()

if __name__ == '__main__':
    # X, y = dataset.read_data()
    # X = dataset.preprocess_data(X, y)

    X, y = dataset.read_data_processed()
    