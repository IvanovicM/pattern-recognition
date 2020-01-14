import numpy as np 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters
from queue import Queue 
from skimage import morphology

def read_data():
    X = []
    y = np.repeat([0, 1, 2, 3, 4], n_examples)
    letter = 'A'
    img_num = 10

    for letter in ['A', 'E', 'I', 'O', 'U']:
        for img_num in range(1, n_examples+1):
            path = 'data/baza{}{}{}{}.bmp'.format(
                letter, img_num // 100, (img_num % 100) // 10, img_num % 10
            )

            img = cv2.imread(path)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(gray_image)
    return X, y

def read_data_processed(n_examples=120):
    X = []
    y = np.repeat([0, 1, 2, 3, 4], n_examples)

    for label in [0, 1, 2, 3, 4]:
        for img_num in range(0, n_examples):
            path = 'data/croped/{}-{}.bmp'.format(label, img_num)

            img = cv2.imread(path)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(gray_image)
    return X, y

def bfs(img, visited, i_start, j_start):
    q = Queue(maxsize = 500)
    q.put([i_start, j_start])
    img[i_start, j_start] = 0 
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

                if img[i_next, j_next] == 255 and not visited[i_next, j_next]:
                    img[i_next, j_next] = 0 
                    visited[i_next, j_next] = True
                    q.put([i_next, j_next])
    return img, visited

def remove_white_edges(img):
    visited = np.zeros(shape=img.shape, dtype=bool)
    for i in [0, img.shape[0] - 1]:
        for j in range(img.shape[1]):
            if not visited[i, j] and img[i, j] == 255:
                img, visited = bfs(img, visited, i, j)
    for i in range(img.shape[0]):
        for j in [0, img.shape[1] - 1]:
            if not visited[i, j] and img[i, j] == 255:
                img, visited = bfs(img, visited, i, j)

    return img

def remove_very_small_components(img):
    nb_components, output, stats, centroids = (
        cv2.connectedComponentsWithStats(img, connectivity=8)
    )
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    min_size = 20
    img_2 = np.zeros((img.shape), dtype=int)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img_2[output == i + 1] = 255
    return img_2

def get_relevant_part(img):
    (h, w) = img.shape

    # JMIN
    jmin = 0
    found = False
    for j in range(w):
        for i in range(h):
            if img[i, j] == 255:
                found = True
                jmin = j
            if found:
                break
        if found:
            break

    # JMAX
    jmax = 0
    found = False
    for j in reversed(range(w)):
        for i in range(h):
            if img[i, j] == 255:
                found = True
                jmax = j
            if found:
                break
        if found:
            break

    # IMIN
    imin = 0
    found = False
    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                found = True
                imin = i
            if found:
                break
        if found:
            break

    # IMAX
    imax = 0
    found = False
    for i in reversed(range(h)):
        for j in range(w):
            if img[i, j] == 255:
                found = True
                imax = i
            if found:
                break
        if found:
            break

    return img[imin:imax, jmin:jmax]

def get_magic_part(img):
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
    img = 255 - img

    img = remove_white_edges(img)
    img = remove_very_small_components(img)
    img = get_relevant_part(img)

    cv2.imwrite('data/croped/{}-{}.bmp'.format(label, img_num), img)

    return img

def preprocess_data(X, y):
    for img_num in range(len(X)):
        img = X[img_num]
        label = y[img_num]
        X[img_num] = preprocess_image(img, np.mod(img_num, n_examples), label)
    
    return X
