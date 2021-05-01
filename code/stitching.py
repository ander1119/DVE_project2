import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def load_images(path):
    images = []
    focal_length = []
    with open(path, 'r') as f:
        for l in f:
            if l[0] == '#':
                continue
            l = l.split()
            if l:
                images.append(cv2.imread(l[0]))
                focal_length.append(float(l[1]))
                print(f'Read {l[0]} {images[-1].shape} with focal length {focal_length[-1]}')
    return images, focal_length

def cylindrical_projection(img, f):
    projection = np.zeros(img.shape, dtype=np.uint8)
    h, w, _ = img.shape
    for x in range(-int(w/2), int(w/2)):
        for y in range(-int(h/2), int(h/2)):
            x_ = f * np.arctan(x / f)
            y_ = f * y / np.sqrt(x ** 2 + f ** 2)
            x_ = round(x_ + w // 2)
            y_ = round(y_ + h // 2)
            if x_ >= 0 and x_ < w and y_ >= 0 and y_ < h:
                projection[y_][x_] = img[y + int(h/2)][x + int(w/2)]

    return projection

def RANSEC(matched_pairs, k=20, threshold=40, n=2):
    best_offset = None
    max_c = -1
    matched_pairs = matched_pairs.tolist()
    for _k in range(k):
        assert len(matched_pairs) > n
        offset = np.array([0, 0])
        sampled_pairs = random.sample(matched_pairs, n)
        for x1, y1, x2, y2 in sampled_pairs:
            offset += np.array([x2 - x1, y2 - y1])
        offset = offset // n
        c = 0
        for pair in matched_pairs:
            diff = np.array([pair[0], pair[1]]) + offset - np.array([pair[2], pair[3]])
            diff = np.dot(diff, diff)
            if diff < threshold:
                c += 1
        if c > max_c:
            max_c = c
            best_offset = offset
    return best_offset

def merge_two_image(img1, img2, offset):
    h, w, _ = img1.shape
    h2, w2, _ = img2.shape
    merged_img = np.zeros((img2.shape[0] + abs(offset[0]), img2.shape[1] + abs(offset[1]), img2.shape[2]), dtype=np.uint8)

    sy = 0 if offset[0] > 0 else -offset[0]
    sx = 0 if offset[1] > 0 else -offset[1]
    merged_img[sy:sy+h2,sx:sx+w2] = img2

    sy = 0 if offset[0] < 0 else offset[0]
    sx = 0 if offset[1] < 0 else offset[1]
    merged_img[sy:sy+h,sx:sx+w] = img1

    # blending
    gg = (offset[1] + w) // 3
    if offset[0] > 0:
        for ii, i in enumerate(range(offset[0], h2)):
            for jj, j in enumerate(range(-offset[1], w)):
                if jj < gg:
                    gamma = 0
                elif jj > gg * 2:
                    gamma = 1
                else:
                    gamma = jj/(w+offset[1])
                i1 = ii
                i2 = ii + offset[0]
                try:
                    merged_img[i][j] = (1 - gamma) * img1[i1][-offset[1] + jj] + gamma * img2[i2][jj]
                except:
                    pass
    else:
        for ii, i in enumerate(range(-offset[0], h)):
            for jj, j in enumerate(range(-offset[1], w)):
                if jj < gg:
                    gamma = 0
                elif jj > gg * 2:
                    gamma = 1
                else:
                    gamma = jj/(w+offset[1])
                i1 = ii - offset[0]
                i2 = ii
                try:
                    merged_img[i][j] = (1 - gamma) * img1[i1][-offset[1] + jj] + gamma * img2[i2][jj]
                except:
                    pass

    return merged_img

