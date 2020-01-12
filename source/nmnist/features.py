import numpy as np 
import matplotlib.pyplot as plt

def get_white_per_image(x):
    f = len([1
             for i in range(x.shape[0])
             for j in range(x.shape[1])
                if x[i, j] == 255])
    return f / (x.shape[0] * x.shape[1])

def feature_left_image_part(x):
    (h, w) = x.shape
    x_part = x[0 : h, 0 : round(w/3)]
    return get_white_per_image(x_part)

def feature_right_image_part(x):
    (h, w) = x.shape
    x_part = x[0 : h, round(2*w/3) : w]
    return get_white_per_image(x_part)

def feature_horizontal_middle_part(x):
    (h, w) = x.shape
    x_part = x[round(3*h/6) : round(5*h/6), 0 : w]
    return get_white_per_image(x_part)

def feature_middle_part(x):
    (h, w) = x.shape
    x_part = x[round(3*h/6) : round(5*h/6), round(3*w/6) : round(5*w/6)]
    return get_white_per_image(x_part)

def get_feature_vector(x):
    f1 = feature_left_image_part(x)
    f2 = feature_right_image_part(x)
    f3 = feature_horizontal_middle_part(x)
    f4 = feature_middle_part(x)

    return [f1, f2, f3, f4]

def make_features(X):
    X_features = []
    for x in X:
        fvector = get_feature_vector(x)
        X_features.append(fvector)
    
    return np.matrix(X_features)