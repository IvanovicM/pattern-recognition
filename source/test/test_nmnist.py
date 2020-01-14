import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from ..nmnist import dataset
from ..nmnist import features
from ..utils import dataplot
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sns.set()
plt.rcdefaults()

def get_new_x(X):
    X_new = []
    for img_num in range(len(X)):
        #resh = gray_image.reshape(200*200)
        img = X[img_num]

        new_1 = [sum(img[:, i]) for i in range(200)]
        new_2 = [sum(img[i, :]) for i in range(200)]

        X_new.append(new_1 + new_2)

    return X_new

def plot_in_2d(X, method='TSNE'):
    n_examples = int(len(X) / 5)
    a_idx = np.arange(0, n_examples)
    e_idx = np.arange(n_examples, 2*n_examples)
    i_idx = np.arange(2*n_examples, 3*n_examples)
    o_idx = np.arange(3*n_examples, 4*n_examples)
    u_idx = np.arange(4*n_examples, 5*n_examples)

    if method == 'TSNE':
        tsne = TSNE(2)  
        X_fit = tsne.fit_transform(X)
    else:
        pca = PCA(2)
        X_fit = pca.fit_transform(X)

    x_a = X_fit[a_idx, 0]
    y_a = X_fit[a_idx, 1]

    x_e = X_fit[e_idx, 0]
    y_e = X_fit[e_idx, 1]

    x_i = X_fit[i_idx, 0]
    y_i = X_fit[i_idx, 1]

    x_o = X_fit[o_idx, 0]
    y_o = X_fit[o_idx, 1]

    x_u = X_fit[u_idx, 0]
    y_u = X_fit[u_idx, 1]

    plt.scatter(x_a, y_a, color='r', label='A')
    plt.scatter(x_e, y_e, color='b', label='E')
    plt.scatter(x_i, y_i, color='m', label='I')
    plt.scatter(x_o, y_o, color='c', label='O')
    plt.scatter(x_u, y_u, color='g', label='U')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    #X, y = dataset.read_data()
    #X = dataset.preprocess_data(X, y)

    print('Read data ...')
    X, y = dataset.read_data_processed(120) 

    print('Make features ...')
    f = features.make_features(X)

    print('Plot 2D ...')
    plot_in_2d(f)