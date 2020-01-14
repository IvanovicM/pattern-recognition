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
    x_part = x[0 : h, 0 : round(w/5)]
    return get_white_per_image(x_part)

def feature_right_image_part(x):
    (h, w) = x.shape
    x_part = x[0 : h, round(4*w/5) : w]
    return get_white_per_image(x_part)

def feature_up_image_part(x):
    (h, w) = x.shape
    x_part = x[0 : round(h/5), 0 : w]
    return get_white_per_image(x_part)

def feature_down_image_part(x):
    (h, w) = x.shape
    x_part = x[round(4*h/5) : h, 0 : w]
    return get_white_per_image(x_part)

def feature_horizontal_middle_part(x):
    (h, w) = x.shape
    x_part = x[round(3*h/5) : round(4*h/5), 0 : w]
    return get_white_per_image(x_part)

def feature_vertical_middle_part(x):
    (h, w) = x.shape
    x_part = x[0 : h, round(3*w/5) : round(4*w/5)]
    return get_white_per_image(x_part)

def feature_middle_part(x):
    (h, w) = x.shape
    x_part = x[round(3*h/5) : round(4*h/5), round(3*w/5) : round(4*w/5)]
    return get_white_per_image(x_part)

def get_feature_vector(x):
    f1 = feature_left_image_part(x)
    f2 = feature_vertical_middle_part(x)
    f3 = feature_right_image_part(x)
    f4 = feature_middle_part(x)
    f5 = feature_up_image_part(x)
    f6 = feature_horizontal_middle_part(x)
    f7 = feature_down_image_part(x)

    return [f1, f2, f3, f4, f5, f6, f7]

def make_features(X):
    X_features = []
    for x in X:
        fvector = get_feature_vector(x)
        X_features.append(fvector)
    
    return np.matrix(X_features)