import cv2 as cv
import numpy as np

import pickle
from sklearn.metrics import accuracy_score
import os

# Load kmeans model
print("Loading Kmeans model...")
filename1 = 'kmeans_model.sav'
k_means = pickle.load(open(filename1, 'rb'))
n_clusters = 1600
# Load SVM model
print("Success")
print("Loading SVM model...")
filename2 = 'gestures_model.sav'
clf = pickle.load(open(filename2, 'rb'))
print("Success")
y_true = []
y_predict = []
menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"


def getThresholdedHand(frame):
    # Convert image to HSV
    hsvim = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Lower boundary of skin color in HSV
    lower = np.array([0, 48, 80], dtype="uint8")
    # Upper boundary of skin color in HSV
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv.inRange(hsvim, lower, upper)

    # Gaussian filter (blur) to remove noise
    skinMask = cv.GaussianBlur(skinMask, (17, 17), 0)

    # get thresholded image
    # ret, thresh1 = cv.threshold(
    # skinMask, 100, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    thresh1 = cv.adaptiveThreshold(
        skinMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 355, 5)

    return thresh1


for g in range(0, 6):
    for i in range(1, 101):
        # Read image
        img = cv.imread(menPath + f'{g}_men ({i}).JPG')
        img = getThresholdedHand(img)

        # Feature extraction
        sift = cv.SIFT_create()
        kp, descriptor = sift.detectAndCompute(img, None)
        # Produce "bag of words" vector
        descriptor = k_means.predict(descriptor)
        print(f"SIFT {g}/{i}")
        vq = [0] * n_clusters
        for feature in descriptor:
            vq[feature] = vq[feature] + 1  # load the model from disk
        y_true.append(g)
        # Predict the result
        y_predict.append(clf.predict([vq]))
for g in range(0, 6):
    for i in range(1, 101):
        # Read image
        img = cv.imread(womenPath + f'{g}_woman ({i}).JPG')
        img = getThresholdedHand(img)
        # Feature extraction
        sift = cv.SIFT_create()
        kp, descriptor = sift.detectAndCompute(img, None)
        # Produce "bag of words" vector
        descriptor = k_means.predict(descriptor)
        print(f"SIFT {g}/{i}")
        vq = [0] * n_clusters
        for feature in descriptor:
            vq[feature] = vq[feature] + 1  # load the model from disk
        y_true.append(g)
        # Predict the result
        y_predict.append(clf.predict([vq]))
accuracy = accuracy_score(y_true, y_predict)
print(f"Accuracy: {accuracy}")
