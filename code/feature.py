import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# some predefined window size
GAUSSIAN = 3
GRADIENT = 3
NON_MAXIMUM = 5
FEATURE_SPACE = 4
HARRIS_K = 0.02
# distance ratio that impacting the number of matching keypoints
MATCHING_RATIO = 0.6

def feature_detect(img):
    img_gray = img
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Ix = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    Iy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    _, ang = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)
    ang = (ang / 45).astype(int)

    Sxx = cv2.GaussianBlur(src=Ix*Ix, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0)
    Sxy = cv2.GaussianBlur(src=Ix*Iy, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0) 
    Syy = cv2.GaussianBlur(src=Iy*Iy, ksize=(GAUSSIAN, GAUSSIAN), sigmaX=0)

    M = np.array([[Sxx, Sxy], [Sxy, Syy]])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    trace = M[0, 0] + M[1, 1]
    
    R = det - HARRIS_K * trace ** 2
    threshold = sorted(R.flatten(), reverse =True)[int(R.flatten().shape[0] / 8)]
    R[R < threshold] = 0
    R = cv2.copyMakeBorder(R, NON_MAXIMUM, NON_MAXIMUM, NON_MAXIMUM, NON_MAXIMUM, cv2.BORDER_REFLECT)

    for i in tqdm(range(NON_MAXIMUM, R.shape[0]-NON_MAXIMUM)):
        for j in range(NON_MAXIMUM, R.shape[1]-NON_MAXIMUM):
            window = R[i-NON_MAXIMUM:i+NON_MAXIMUM+1, j-NON_MAXIMUM:j+NON_MAXIMUM+1]
            maximum = np.max(window)
            window[window < maximum] = 0
            R[i-NON_MAXIMUM:i+NON_MAXIMUM+1, j-NON_MAXIMUM:j+NON_MAXIMUM+1] = window

    R = R[NON_MAXIMUM:-NON_MAXIMUM, NON_MAXIMUM:-NON_MAXIMUM]
    keypoints = np.argwhere(R>0)

    mag_ext = cv2.copyMakeBorder((Ix**2+Iy**2)**(1/2), FEATURE_SPACE, FEATURE_SPACE, FEATURE_SPACE, FEATURE_SPACE, cv2.BORDER_REFLECT)
    ang_ext = cv2.copyMakeBorder(ang, FEATURE_SPACE, FEATURE_SPACE, FEATURE_SPACE, FEATURE_SPACE, cv2.BORDER_REFLECT)
    features = np.zeros((keypoints.shape[0], 32))
    for idx, p in enumerate(keypoints):
        newx, newy = p[0] + FEATURE_SPACE, p[1] + FEATURE_SPACE
        LU = np.bincount(ang_ext[newx-FEATURE_SPACE:newx, newy-FEATURE_SPACE:newy].flatten(), weights=mag_ext[newx-FEATURE_SPACE:newx, newy-FEATURE_SPACE:newy].flatten(), minlength=8)
        RU = np.bincount(ang_ext[newx+1:newx+FEATURE_SPACE+1, newy-FEATURE_SPACE:newy].flatten(), weights=mag_ext[newx+1:newx+FEATURE_SPACE+1, newy-FEATURE_SPACE:newy].flatten(), minlength=8)
        LD = np.bincount(ang_ext[newx-FEATURE_SPACE:newx, newy+1:newy+FEATURE_SPACE+1].flatten(), weights=mag_ext[newx-FEATURE_SPACE:newx, newy+1:newy+FEATURE_SPACE+1].flatten(), minlength=8)
        RD = np.bincount(ang_ext[newx+1:newx+FEATURE_SPACE+1, newy+1:newy+FEATURE_SPACE+1].flatten(), weights=mag_ext[newx+1:newx+FEATURE_SPACE+1, newy+1:newy+FEATURE_SPACE+1].flatten(), minlength=8)
        feat_vec = np.concatenate((LU, RU, LD, RD))
        normalized_feat_vec = feat_vec / np.linalg.norm(feat_vec)
        features[idx] = normalized_feat_vec
    # print(features[0].shape)
    # figure, axes = plt.subplots()
    # plt.imshow(img)
    # for x, y, _1, _2 in features:
    #     draw_circle = plt.Circle((y, x), 3)
    #     axes.add_artist(draw_circle)
    # plt.show()

    return keypoints, features

def feature_match(img1, img2):
    keypoints1, feat1 = feature_detect(img1)
    keypoints2, feat2 = feature_detect(img2)
    coordinate = []
    for i, f in enumerate(feat1):
        min_val = np.partition(np.linalg.norm(feat2 - f, axis=1), 0)[0]
        min_arg = np.argpartition(np.linalg.norm(feat2 - f, axis=1), 0)[0]
        sec_min_val = np.partition(np.linalg.norm(feat2 - f, axis=1), 1)[1]
        sec_min_arg = np.argpartition(np.linalg.norm(feat2 - f, axis=1), 1)[1]
        if min_val / sec_min_val < MATCHING_RATIO:
            coordinate.append([keypoints1[i][0], keypoints1[i][1], keypoints2[min_arg][0], keypoints2[min_arg][1]])
    print("got {} matches".format(len(coordinate)))

    return np.array(coordinate)

def draw_matches(img1, img2, coordinate):
    width = img1.shape[1] + img2.shape[1]
    height = max(img1.shape[0], img2.shape[0])
    coordinate[:, 3] += img1.shape[1]
    img_concat = np.zeros((height, width, 3), dtype=np.uint8)
    img_concat[0:img1.shape[0], 0:img2.shape[1], :] = img1
    img_concat[0:img2.shape[0], img1.shape[1]:, :] = img2

    figure, axes = plt.subplots()
    plt.imshow(img_concat)
    for x1, y1, x2, y2 in coordinate:
        circle1 = plt.Circle((y1, x1), 2)
        axes.add_artist(circle1)
        circle2 = plt.Circle((y2, x2), 2)
        axes.add_artist(circle2)
        plt.plot([y1, y2], [x1, x2], marker='o')
    plt.show()

sharpen_kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

def sharpen(img):
    return cv2.filter2D(img, -1, sharpen_kernel)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str)
    parser.add_argument('img2', type=str)
    args = parser.parse_args()

    img1 = sharpen(cv2.imread(args.img1))
    img2 = sharpen(cv2.imread(args.img2))

    coordinate = feature_match(img1, img2)
    draw_matches(img1, img2, coordinate)