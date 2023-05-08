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
        print(thresh1.shape)
        return thresh1
    @staticmethod
    def getGuassianThresholdedHand(frame):
        # frame = cv.GaussianBlur(frame, (41, 41), 0)
        alpha = [[0.298936021293775390],
                 [0.587043074451121360],
                 [0.140209042551032500]]
        print(f"frame shape {frame.shape}")
        binaryFame = np.zeros((frame.shape[0], frame.shape[1]))
        print(f"binaryFame shape {binaryFame.shape}")
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.normalize(frame, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32FC2)

        # thres = np.array(())
        Ix = frame @ alpha
        IxHat = np.zeros((frame.shape[0], frame.shape[1]))
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                IxHat = max(frame[i,j,1],frame[i,j,2])
                # if frame[i,j,1] > frame[i,j,2]:
                #     IxHat = 0
                # else:
                #     IxHat = 1

        Ex = Ix - IxHat
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if Ex[i,j] <= 0.1177 and Ex[i,j] >= 0.02511:
                    binaryFame[i,j] = 255
                else:
                    binaryFame[i,j] = 0
        return binaryFame
    @staticmethod
    def skin_color_thresholding(image):
        # image = cv.GaussianBlur(image, (13, 13), 0)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        alpha = np.array([0.298936021293775390, 0.587043074451121360, 0.140209042551032500]).T
        c = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        I = np.dot(c, alpha)
        IHat = np.maximum(c[:,:, 1], c[:,:, 2])
        e = I - IHat
        fskin = np.logical_and(e >= 0.02711, e <= 0.1177)
        return fskin.astype(int)


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
