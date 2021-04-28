import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

GAUSSIAN = 3
GRADIENT = 3
NON_MAXIMUM = 2
HARRIS_K = 0.04

def feature_detect(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Ix = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0, ksize=GRADIENT)
    Iy = cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1, ksize=GRADIENT)

    Sxx = cv2.GaussianBlur(src=Ix*Ix, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0)
    Sxy = cv2.GaussianBlur(src=Ix*Iy, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0) 
    Syy = cv2.GaussianBlur(src=Iy*Iy, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0)

    M = np.array([[Sxx, Sxy], [Sxy, Syy]])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    trace = H[0, 0] + H[1, 1]
    
    R = det - HARRIS_K * trace ** 2
    threshold = sorted(R.flatten(), reverse =True)[int(R.flatten().shape[0]) / 4]
    R[R < threshold] = 0
    R = cv2.copyMakeBorder(R, NON_MAXIMUM, NON_MAXIMUM, NON_MAXIMUM, NON_MAXIMUM, cv2.BORDER_REFLECT)

    features = []
    for i in range(NON_MAXIMUM, R.shape[0]-NON_MAXIMUM):
        for j in range(NON_MAXIMUM, R.shape[1]-NON_MAXIMUM):
            window = R[i-NON_MAXIMUM:i+NON_MAXIMUM+1 : j-NON_MAXIMUM, j+NON_MAXIMUM+1]
            max = np.max(R)
            window[w < window] = 0
            R[i-NON_MAXIMUM:i+NON_MAXIMUM+1 : j-NON_MAXIMUM, j+NON_MAXIMUM+1] = window
            features.append([i-NON_MAXIMUM, j-NON_MAXIMUM, max])
    R = R[NON_MAXIMUM:-NON_MAXIMUM, NON_MAXIMUM:-NON_MAXIMUM]

    


if __name__ == '__main__':
    img = cv2.imread("test_data/denny/denny00.jpg")
    feature_detect(img)