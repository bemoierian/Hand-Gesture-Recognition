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
        # print(f"fskin shape {fskin.shape}")
        return fskin.astype(int)


    @staticmethod
    def getMaskedHand(frame):
        # # ------------Thresholding method 1-----------------
        # # Convert image to HSV
        # hsvim = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # # Lower boundary of skin color in HSV
        # # lower = np.array([0, 20, 70], dtype="uint8")
        # lower = np.array([0, 48, 80], dtype="uint8")
        # # Upper boundary of skin color in HSV
        # upper = np.array([20, 255, 255], dtype="uint8")
        # skinMask = cv.inRange(hsvim, lower, upper)
        # # Gaussian filter (blur) to remove noise
        # skinMask = cv.GaussianBlur(skinMask, (17, 17), 0)

        # ------------Thresholding method 2-----------------
        threshImg = Utils.skin_color_thresholding(frame)
        threshImg = cv.normalize(threshImg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # print(f"threshImg shape {threshImg.shape}")
        # print(f"threshImg: {threshImg}")


        greyImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # print(f"greyImg shape {greyImg.shape}")
        masked = cv.bitwise_and(greyImg, greyImg, mask=threshImg)
        # maskedBlurred = cv.GaussianBlur(masked, (41, 41), 0)


        return masked

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
        # filename2 = 'gestures_model.sav'
        filename2 = 'Hog_model.sav'
        clf = pickle.load(open(filename2, 'rb'))
        return clf
    @staticmethod
    def loadPCAModel():
        # Load PCA model
        print("Loading PCA model...")
        filename2 = 'pca.sav'
        pca = pickle.load(open(filename2, 'rb'))
        return pca

    @staticmethod
    def extractInteger(filename):
        # print(filename)
        k = 0
        try:
            k = int(filename.split('.')[0].split('(')[1].split(')')[0])
        except:
            k = 0
        return k
    @staticmethod
    def adjust_image(image):
        # Convert image to LAB color space
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)

        # Split the LAB image into different channels
        l, a, b = cv.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
        cl = clahe.apply(l)

        # Merge the CLAHE enhanced L channel with the original A and B channels
        limg = cv.merge((cl,a,b))

        # Convert the LAB image back to BGR color space
        final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
        return final
    @staticmethod
    def gamma_correction(image, gamma=1.0):
        # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # Apply gamma correction using the lookup table
        corrected_image = cv.LUT(image, table)

        return corrected_image