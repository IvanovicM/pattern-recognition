import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

# h_max = 385, w_max = 436
# h_mim = 349, w_mim = 391

def read_data():
    n_examples = 120 * 5
    X = []

    for img_num in range(1, 121):
        for letter in ['A', 'E', 'I', 'O', 'U']:
            path = 'data/baza{}{}{}{}.bmp'.format(
                letter, img_num // 100, (img_num % 100) // 10, img_num % 10
            )

            img = Image.open(path).convert('LA')
            X.append(img)
    
    return X
