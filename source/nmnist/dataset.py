import numpy as np 
import cv2
from PIL import Image

# h_max = 385, w_max = 436
# h_min = 349, w_min = 391

def read_data():
    n_examples = 120 * 5
    X = []
    y = np.repeat([0, 1, 2, 3, 4], 120)

    for letter in ['A', 'E', 'I', 'O', 'U']:
        for img_num in range(1, 121):
            path = 'data/baza{}{}{}{}.bmp'.format(
                letter, img_num // 100, (img_num % 100) // 10, img_num % 10
            )

            pil_img = Image.open(path).convert('RGB')
            img = np.array(pil_img)
            X.append(img)
    
    return X, y

def preprocess_data(X):
    for img_num in range(len(X)):
        img = X[img_num]
        img = img[0 : 340, 0 : 390]
        X[img_num] = img
    
    return X
