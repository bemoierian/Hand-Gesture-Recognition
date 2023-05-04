import numpy as np
import cv2 as cv
import pickle


class Utils:
    @staticmethod
    def getThresholdedHand(frame):
        frame = cv.GaussianBlur(frame, (41, 41), 0)
        # Convert image to HSV
        hsvim = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # Lower boundary of skin color in HSV
        lower = np.array([0, 48, 80], dtype="uint8")
        # Upper boundary of skin color in HSV
        upper = np.array([120, 255, 255], dtype="uint8")
        # upper = np.array([20, 255, 255], dtype="uint8")
        skinMask = cv.inRange(hsvim, lower, upper)

        # Gaussian filter (blur) to remove noise
        # skinMask = cv.GaussianBlur(skinMask, (71, 71), 0)
        skinMask = cv.GaussianBlur(skinMask, (71, 71), 0)

        # get thresholded image
        ret, thresh1 = cv.threshold(
            skinMask, 100, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # thresh1 = cv.adaptiveThreshold(
        #     skinMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 355, 5)
        # thresh1 = cv.bitwise_and(image, mask)

        kernel = np.ones((10,10),np.uint8)
        # thresh1 = cv.dilate(thresh1,kernel,iterations = 5)
        thresh1 = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
        return thresh1
    @staticmethod
    def getMaskedHand(frame):
        # Convert image to HSV
        hsvim = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # Lower boundary of skin color in HSV
        # lower = np.array([0, 20, 70], dtype="uint8")
        lower = np.array([0, 48, 80], dtype="uint8")
        # Upper boundary of skin color in HSV
        upper = np.array([20, 255, 255], dtype="uint8")
        skinMask = cv.inRange(hsvim, lower, upper)

        # Gaussian filter (blur) to remove noise
        skinMask = cv.GaussianBlur(skinMask, (17, 17), 0)
        greyImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        masked = cv.bitwise_and(greyImg, greyImg, mask=skinMask)
        maskedBlurred = cv.GaussianBlur(masked, (41, 41), 0)

        return maskedBlurred

    @staticmethod
    def loadKmeansModel():
        # Load kmeans model
        print("Loading Kmeans model...")
        filename1 = 'kmeans_model.sav'
        k_means = pickle.load(open(filename1, 'rb'))
        return k_means

    @staticmethod
    def loadSVMModel():
        # Load SVM model
        print("Loading SVM model...")
        filename2 = 'gestures_model.sav'
        clf = pickle.load(open(filename2, 'rb'))
        return clf

    @staticmethod
    def extractInteger(filename):
        # print(filename)
        k = 0
        try:
            k = int(filename.split('.')[0].split('(')[1].split(')')[0])
        except:
            k = 0
        return k
