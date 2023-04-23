import os
import cv2 as cv
from utils import Utils

menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"
outputPath = "../thresholded_images/"

label = 0
for i in range(1, 20):
    class_dir = os.path.join(menPath, f"{label}")
    imgPath = os.path.join(class_dir, f'{label}_men ({i}).JPG')
    img = cv.imread(imgPath)
    img = Utils.getThresholdedHand(img)
    outPath = os.path.join(outputPath, f'{label}_men ({i}).JPG')
    cv.imwrite(outPath, img)
    # if cv.waitKey(1) & 0xff == 27:
    #     break
