import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from ..nmnist import dataset
from ..utils import dataplot

sns.set()
plt.rcdefaults()

if __name__ == '__main__':
    X, y = dataset.read_data()
    X = dataset.preprocess_data(X)
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)