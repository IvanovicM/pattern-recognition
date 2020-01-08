import numpy as np
from sklearn.utils import shuffle
from ..utils import datagen
from ..clustering import kmeans
from ..clustering import squarerr

def get_data():
    N = 500

    # Class 0
    c_x1 = 5.5
    c_x2 = 7.5
    angle = 0.25
    dist = 1.5
    x1 = datagen.generate_uniform_doughnut_part(c_x1, c_x2, angle, dist, N)

    # Class 1
    c_x1 = 6
    c_x2 = 6.5
    x2 = datagen.generate_uniform_circle(c_x1, c_x2, N)

    data = np.concatenate((x1, x2), axis=0)
    return shuffle(data)

if __name__ == '__main__':
    # Generate 2 non-linear separable classes
    data = get_data()

    # Try kmeans (won't work well because classes are non-linear separable)
    assignments, centers, iters = kmeans.kmeans(data, 2)

    # Try square error clustering (suitable for non-linear separable classes)
    assignments, iters = squarerr.square_error(data, 2)