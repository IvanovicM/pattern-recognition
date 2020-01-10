import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from ..nmnist import dataset
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

def experiments(X):
    X_new = get_new_x(X) 

    a_idx = np.arange(0, 120)
    e_idx = np.arange(120, 240)
    i_idx = np.arange(240, 360)
    o_idx = np.arange(360, 480)
    u_idx = np.arange(480, 600)

    # # TSNE
    # tsne = TSNE(2)  
    # X_fit = tsne.fit_transform(X_new)

    # PSA
    pca = PCA(2)
    X_fit = pca.fit_transform(X_new)

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

    print(len(x_a))
    print(len(x_e))
    print(len(x_i))
    print(len(x_o))
    print(len(x_u))

    plt.scatter(x_a, y_a, color='r', label='A')
    plt.scatter(x_e, y_e, color='b', label='E')
    plt.scatter(x_i, y_i, color='m', label='I')
    plt.scatter(x_o, y_o, color='c', label='O')
    plt.scatter(x_u, y_u, color='g', label='U')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # X, y = dataset.read_data()
    # X = dataset.preprocess_data(X, y)

    X, y = dataset.read_data_processed() 
    experiments(X)

