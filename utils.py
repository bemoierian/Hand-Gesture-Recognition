import numpy as np
import cv2 as cv
import pickle
import time
from sklearn.mixture import GaussianMixture

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
        filename2 = 'SVM_model.sav'
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
    def extractInteger2(filename):
        # print(filename)
        k = 0
        try:
            k = int(filename.split('.')[0])
        except:
            k = 0
        return k
    @staticmethod
    def adjust_image(image):
        # Convert image to LAB color space
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        # lab = cv.cvtColor(image, cv.COLOR_RGB2LAB)

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
    
    @staticmethod
    def thresholdingUsingMorph(img):
        # convert the image from BGR to ycrcb
        ycrcbImg = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        # get the ycrcb channels
        y, cr, cb = cv.split(ycrcbImg)
        # apply the gaussian filter to remove noise
        gaussImg = cv.GaussianBlur(cr, (5, 5), 0)
        # apply the threshold to get the mask
        _, mask = cv.threshold(gaussImg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # apply the median filter to remove noise
        mask = cv.medianBlur(mask, 5)
        # apply the closing to fill the holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # apply the opening to remove noise
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # apply the dilation to get the mask
        mask = cv.dilate(mask, kernel, iterations=1)
        # apply the erosion to get the mask
        mask = cv.erode(mask, kernel, iterations=1)
        # apply the median filter to remove noise
        mask = cv.medianBlur(mask, 5)
        # apply the closing to fill the holes
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # apply the opening to remove noise
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # apply the dilation to get the mask
        mask = cv.dilate(mask, kernel, iterations=1)
        # apply the erosion to get the mask
        mask = cv.erode(mask, kernel, iterations=1)
        # apply the median filter to remove noise
        mask = cv.medianBlur(mask, 5)
        # apply the closing to fill the holes
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # apply the opening to remove noise
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # apply the dilation to get the mask
        mask = cv.dilate(mask, kernel, iterations=1)
        # apply the erosion to get the mask
        mask = cv.erode(mask, kernel, iterations=1)
        # apply the median filter to remove noise
        mask = cv.medianBlur(mask, 5)
        # apply the closing to fill the holes
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # apply the opening to remove noise
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # apply the dilation to get the mask
        mask = cv.dilate(mask, kernel, iterations=1)
        # apply the erosion to get the mask
        mask = cv.erode(mask, kernel, iterations=1)
        # apply the median filter to remove noise
        mask = cv.medianBlur(mask, 5)
        # apply the closing to fill the holes
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # apply the opening to remove noise
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # apply the dilation to get the mask
        mask = cv.dilate(mask, kernel, iterations=1)
        # apply the erosion to get the mask
        mask = cv.erode(mask, kernel, iterations=1)
        # apply the median filter to remove noise
        mask = cv.medianBlur(mask, 5)
        # apply the closing to fill the holes
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # apply the opening to remove noise
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # apply the dilation to get the mask
        mask = cv.dilate(mask, kernel, iterations=1)
        # apply the erosion to get the mask
        mask = cv.erode(mask, kernel, iterations=1)
        # apply the median filter to remove noise
        mask = cv.medianBlur(mask, 5)
        # apply the closing to fill the holes
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # apply the opening to remove noise
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        return mask
    
    @staticmethod
    def removeShadows(frame):
        # Convert image to Lab
        rgbImg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        labImg = cv.cvtColor(rgbImg, cv.COLOR_RGB2LAB)
        # The values of L,A and B channels are  mapped  in  the  range  from  0  up  to  255 
        meanL = np.mean(labImg[:, :, 0])
        meanA = np.mean(labImg[:, :, 1])
        meanB = np.mean(labImg[:, :, 2])
        stdL = np.std(labImg[:, :, 0])
        # stdB = np.std(labImg[:, :, 2])

        # if meanA + meanB <=256:#outdoor
        #         print("first")
        #         mask =labImg[:, :, 0] <= ( meanL - stdL/3)
        #         labImg[mask] = np.array([255, 128, 128])
        #         labImg[(~mask)] = np.array([0, 128, 128])
        # else:#indoor
        #         print("second")
        #         mask1 = labImg[:, :, 0] <=  ( meanL )
        #         mask2 = labImg[:,:, 2] <=  (meanB)
        #         mask = mask1 & mask2
        #         labImg[mask] = np.array([255, 128, 128])#white
        #         labImg[(~mask)] = np.array([0, 128, 128])#black
        
        mask =labImg[:, :, 0] <= ( meanL - stdL/3)
        labImg[mask] = np.array([255, 128, 128])
        labImg[(~mask)] = np.array([0, 128, 128])
        rgbImg2 = cv.cvtColor(labImg, cv.COLOR_LAB2RGB)
        NormalizedImg = cv.normalize(rgbImg2, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX).astype(np.float32)
        NormalizedImg2 = cv.normalize(rgbImg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX).astype(np.float32)

        kernel = np.ones((13, 13), np.uint8)
        img_dilation = cv.dilate(NormalizedImg, kernel, iterations=1)
        img_erosion = cv.erode(img_dilation, kernel, iterations=1)
        # Gaussian filter (blur) to remove noise
        gaussImg = cv.GaussianBlur(img_erosion, (13, 13), 0)
        # get the white and black pixels 
        white_pixels = (gaussImg == np.array([1, 1, 1])).all(axis=2)
        black_pixels = (gaussImg == np.array([0, 0, 0])).all(axis=2)
        # get the average of the white and black pixels in the original img
        # avgR_shadow = np.mean(NormalizedImg2[:,:,0][white_pixels])
        # avgG_shadow = np.mean(NormalizedImg2[:,:,1][white_pixels])
        # avgB_shadow = np.mean(NormalizedImg2[:,:,2][white_pixels])
        # avgR_without = np.mean(NormalizedImg2[:,:,0][black_pixels])
        # avgG_without = np.mean(NormalizedImg2[:,:,1][black_pixels])
        # avgB_without = np.mean(NormalizedImg2[:,:,2][black_pixels])

        avgR_shadow = 0
        avgG_shadow = 0
        avgB_shadow = 0
        avgR_without = 0
        avgG_without = 0
        avgB_without = 0

        # loop over the image with the kernel
        for i in range(0, NormalizedImg2.shape[0], 21):
                for j in range(0, NormalizedImg2.shape[1], 21):
                        # get the average of the white and black pixels in the original img
                        if gaussImg[i:i+21, j:j+21, 0][white_pixels[i:i+21, j:j+21]].shape[0] >=200:
                                avgR_shadow = np.mean(NormalizedImg2[i:i+21, j:j+21, 0][white_pixels[i:i+21, j:j+21]])
                                avgG_shadow = np.mean(NormalizedImg2[i:i+21, j:j+21, 1][white_pixels[i:i+21, j:j+21]])
                                avgB_shadow = np.mean(NormalizedImg2[i:i+21, j:j+21, 2][white_pixels[i:i+21, j:j+21]])
                                avgR_without = np.mean(NormalizedImg2[i:i+21, j:j+21, 0][black_pixels[i:i+21, j:j+21]])
                                avgG_without = np.mean(NormalizedImg2[i:i+21, j:j+21, 1][black_pixels[i:i+21, j:j+21]])
                                avgB_without = np.mean(NormalizedImg2[i:i+21, j:j+21, 2][black_pixels[i:i+21, j:j+21]])
                                # print("all",NormalizedImg2[i:i+21, j:j+21, 0][white_pixels[i:i+21, j:j+21]].shape)
                                # the ratio between the average of the shadow region and the average of the non-shadow region
                                ratioR =avgR_shadow/avgR_without
                                ratioG = avgG_shadow / avgG_without
                                ratioB = avgB_shadow / avgB_without
                                # print(ratioR, ratioG, ratioB)
                                NormalizedImg2[i:i+21, j:j+21][white_pixels[i:i+21, j:j+21]] *= np.array([ratioR, ratioG, ratioB])
                                NormalizedImg2[i:i+21, j:j+21] = cv.medianBlur(NormalizedImg2[i:i+21, j:j+21], 5)
        return NormalizedImg2


    @staticmethod
    def extract_hand(img,flag,img_width=256):
        # Load the image
        # start_time = ti   me.time()
        h, w = img.shape[:2]
        if flag:
            new_height = int(h * img_width / w)
        else:
            new_height = 72
        img_size = (img_width, new_height)
        img = cv.resize(img, img_size)    # resize image

        # Convert the image from BGR to YCbCr
        ycbcr_image = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

        # Extract Cr and Cb channels
        cr = ycbcr_image[:, :, 1]
        cb = ycbcr_image[:, :, 2]

        # Reshape channels into a 1D array
        cr_flat = cr.flatten().reshape(-1, 1)
        cb_flat = cb.flatten().reshape(-1, 1)

        # Concatenate channels into a single array
        x = np.concatenate((cr_flat, cb_flat), axis=1)

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=2)
        gmm.fit(x)

        # Predict probabilities of each pixel belonging to background or foreground
        probs = gmm.predict_proba(x)
        probs_bg = probs[:, 0].reshape(cr.shape)
        probs_fg = probs[:, 1].reshape(cr.shape)

        # Threshold probabilities to extract foreground object from background
        thresh = 0.5
        mask = (probs_fg > thresh).astype(np.uint8) * 255

        # Apply mask to original image
        result1 = cv.bitwise_and(img, img, mask=mask)
        result2 = cv.bitwise_and(img, img, mask=~(mask))

        # Convert image to YCrCb color space
        img_ycrcb1 = cv.cvtColor(result1, cv.COLOR_BGR2YCrCb)
        img_ycrcb2 = cv.cvtColor(result2, cv.COLOR_BGR2YCrCb)

        # Define range of skin color in YCrCb color space
        lower_skin = np.array([[[0, 135, 85]]], dtype=np.uint8)
        upper_skin = np.array([[[255, 180, 135]]], dtype=np.uint8)
        typicalSkinColorYCrCb = np.mean([lower_skin, upper_skin], axis=0)

        l1 = gmm.predict(typicalSkinColorYCrCb[0,0,1:].reshape(1, -1))
        l2 = gmm.predict(lower_skin[0,0,1:].reshape(1, -1))
        l3 = gmm.predict(upper_skin[0,0,1:].reshape(1, -1))

        if (l1[0]==1 and l2[0]==1) and l3[0]==1:
            result = result1
        else:
            result = result2
        # Reshape the image to a 2D array of pixels and 3 color values (RGB)
        rows, cols = result.shape[0], result.shape[1]
        image_2d = result.reshape(rows*cols, 3)

        # Perform k-means clustering with 3 clusters using OpenCV
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv.kmeans(image_2d.astype(np.float32), 2, None, criteria, 10, flags)

        # Reshape the labels back into the original image shape
        labels_2d = labels.reshape(rows, cols)

        # Get the pixel indices for each label
        indices_0 = (labels_2d == 0)
        indices_1 = (labels_2d == 1)
        # Get a separate image for each cluster
        result_0 = result.copy()
        result_0[~indices_0] = 0

        result_1 = result.copy()
        result_1[~indices_1] = 0

        mean_0 = np.mean(result_0)
        mean_1 = np.mean(result_1)
        # print(mean_0, mean_1)
        if mean_1 > mean_0:
            result_temp = result_1
            # indices_temp = indices_1
        else:
            result_temp = result_0
            # indices_temp = indices_0
        #  Create a mask for the hand using the pixel indices of one of the clusters
        # mask = np.zeros((rows, cols), dtype=bool)
        # mask[indices_temp] = True

        # # Apply the mask to the original image to get only the hand
        # hand_image = np.zeros_like(result)
        # hand_image[mask] = result[mask]
        # end_time = time.time()
        # takenTime = end_time - start_time
        # print("Time taken: ", takenTime)
        return result_temp
    
    @staticmethod
    def convert_58ulbp_to_9ulbp(code):
        # Convert the code to binary
        binary_code = bin(int(code))[2:].zfill(58)
        
        # Split the binary code into 9 parts of 6 bits each
        parts = [binary_code[i:i+6] for i in range(0, len(binary_code), 6)]
        
        # Convert each part to decimal and concatenate them
        decimal_code = ''.join([str(int(part, 2)) for part in parts])
        
        # Convert the decimal code to integer
        return int(decimal_code)
        
    @staticmethod
    def get_9ULBP(image):
        height, width = image.shape[:2]
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i,j]
                code = 0
                code |= (image[i-1,j-1] > center) << 7
                code |= (image[i-1,j] > center) << 6
                code |= (image[i-1,j+1] > center) << 5
                code |= (image[i,j+1] > center) << 4
                code |= (image[i+1,j+1] > center) << 3
                code |= (image[i+1,j] > center) << 2
                code |= (image[i+1,j-1] > center) << 1
                code |= (image[i,j-1] > center) << 0
                lbp[i-1,j-1] = Utils.convert_58ulbp_to_9ulbp(code)
                # lbp[i-1,j-1] = code
        hist = cv.calcHist([lbp], [0], None, [9], [0, 9])
        # hist = cv.calcHist([lbp], [0], None, [58], [0, 58])
        hist = cv.normalize(hist, hist).flatten()
        return hist
