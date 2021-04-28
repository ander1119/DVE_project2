import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

GAUSSIAN = 3
GRADIENT = 3

def feature_detect(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Ix = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0, ksize=GRADIENT)
    Iy = cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1, ksize=GRADIENT)

    Sxx = cv2.GaussianBlur(src=Ix*Ix, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0)
    Sxy = cv2.GaussianBlur(src=Ix*Iy, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0) 
    Syy = cv2.GaussianBlur(src=Iy*Iy, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0)

    M = np.array([[Sxx, Sxy], [Sxy, Syy]])
    tmp = np.swapaxes(M, 0, 2)
    tmp = np.swapaxes(tmp, 1, 3)
    print(tmp[0, 0])
    print(Sxx[0, 0])
    print(Sxy[0, 0])
    print(Syy[0, 0])

    print(M.shape)


if __name__ == '__main__':
    img = cv2.imread("test_data/denny/denny00.jpg")
    feature_detect(img)