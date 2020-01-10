import numpy as np 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters
from queue import Queue 

n_examples = 120

def read_data():
    X = []
    y = np.repeat([0, 1, 2, 3, 4], n_examples)

    for letter in ['A', 'E', 'I', 'O', 'U']:
        for img_num in range(1, n_examples+1):
            path = 'data/baza{}{}{}{}.bmp'.format(
                letter, img_num // 100, (img_num % 100) // 10, img_num % 10
            )

            img = cv2.imread(path)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(gray_image)
    return X, y

def read_data_processed():
    X = []
    y = np.repeat([0, 1, 2, 3, 4], n_examples)

    for label in [0, 1, 2, 3, 4]:
        for img_num in range(0, n_examples):
            path = 'data/processed/{}-{}.bmp'.format(label, img_num)

            img = cv2.imread(path)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(gray_image)
    return X, y

def bfs(img, visited, i_start, j_start):
    q = Queue(maxsize = 500)
    q.put([i_start, j_start])
    img[i_start, j_start] = 255 
    visited[i_start, j_start] = True

    while not q.empty():
        [i_curr, j_curr] = q.get()
        dij = [-1, 0, 1]

        for di in dij:
            i_next = i_curr + di
            if i_next < 0 or i_next >= img.shape[0]:
                    continue
            for dj in dij:
                j_next = j_curr + dj
                if j_next < 0 or j_next >= img.shape[1]:
                    continue
                if (i_next > 40 and i_next < img.shape[0] - 40) and (
                    j_next > 40 and j_next < img.shape[1] - 40):
                   continue

                if img[i_next, j_next] == 0 and not visited[i_next, j_next]:
                    img[i_next, j_next] = 255 
                    visited[i_next, j_next] = True
                    q.put([i_next, j_next])
    return img, visited

def remove_black_edges(img):
    visited = np.zeros(shape=img.shape, dtype=bool)
    for i in [0, img.shape[0] - 1]:
        for j in range(img.shape[1]):
            if not visited[i, j] and img[i, j] == 0:
                img, visited = bfs(img, visited, i, j)
    for i in range(img.shape[0]):
        for j in [0, img.shape[1] - 1]:
            if not visited[i, j] and img[i, j] == 0:
                img, visited = bfs(img, visited, i, j)

    return img

def get_relevant_part(img):
    x, y, w, h = cv2.boundingRect(255 - img)
    MAGIC = 200

    cx = int(x + w / 2)
    cy = int(y + h / 2)

    w, h = img.shape
    rx = cx - MAGIC / 2
    ry = cy - MAGIC / 2
    rw = MAGIC
    rh = MAGIC

    if rx + rw > w:
        rw = w - rx
    if rx < 0:
        rx = 0
    if ry + rh > h:
        rh = h - ry
    if ry < 0:
        ry = 0

    rx = int(rx)
    ry = int(ry)
    rw = int(rw)
    rh = int(rh)

    img = img[ry : ry+rh, rx : rx+rw]
    if img.shape[0] != MAGIC or img.shape[1] != MAGIC:
        new_img = np.ones((MAGIC, MAGIC)) * 255
        new_img[0 : img.shape[0], 0 : img.shape[1]] = img
        img = new_img
    return img

def preprocess_image(img, img_num, label):
    print('Processing image {} of label {}...'.format(img_num, label))

    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    img = remove_black_edges(img)
    img = get_relevant_part(img)    
    cv2.imwrite('data/processed/{}-{}.bmp'.format(label, img_num), img)

    return img

def preprocess_data(X, y):
    for img_num in range(len(X)):
        img = X[img_num]
        label = y[img_num]
        X[img_num] = preprocess_image(img, np.mod(img_num, n_examples), label)
    
    return X
