import os
import cv2 as cv
from utils import Utils
import numpy as np

menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"
outputPath = "../thresholded_images/"
sift = cv.SIFT_create()

label = 2
for i in range(8, 15):
    class_dir = os.path.join(menPath, f"{label}")
    imgPath = os.path.join(class_dir, f'{label}_men ({i}).JPG')
    img = cv.imread(imgPath)
    # img = cv.normalize(img, None, 0, 255,
    #                                    cv.NORM_MINMAX).astype('uint8')
    img = Utils.getThresholdedHand(img)
    # cv.GaussianBlur(img, (41, 41), 0)
    # img = Utils.getMaskedHand(img) 
    # kp, descriptor = sift.detectAndCompute(img, None)
    # print(f"descriptor shape {descriptor.shape}")
    # img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
    # t_lower = 0  # Lower Threshold
    # t_upper = 255  # Upper threshold
    # edge = cv.Canny(img, t_lower, t_upper)

    kernel = np.ones((20,20),np.uint8)
    img = cv.dilate(img,kernel,iterations = 5)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    kp, descriptor = sift.detectAndCompute(img, None)
    # print(f"descriptor shape {descriptor.shape}")
    img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)


    outPath = os.path.join(outputPath, f'{label}_men ({i}).JPG')
    cv.imwrite(outPath, img2)
    # if cv.waitKey(1) & 0xff == 27:
    #     break
