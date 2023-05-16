import os
import cv2 as cv
from utils import Utils
import numpy as np

# menPath = "../resized/men/"
# womenPath = "../resized/Women/"
# menPath = "../Dataset_0-5/men/"
menPath = "data"
# womenPath = "../Dataset_0-5/Women/"
testImgPath = "../"
outputPath = "../thresholded_images/"
# sift = cv.SIFT_create()
# Set desired image size
img_width = 120
label = 2
for i in range(1, 55):
    class_dir = os.path.join(os.getcwd(),menPath)
    imgPath = os.path.join(class_dir, f' ({i}).jpg')
    img = cv.imread(imgPath)
    # img = Utils.gamma_correction(img, 0.5)
    # img = Utils.adjust_image(img)
    img = Utils.getMaskedHand(img)
    h, w = img.shape[:2]
    # new_height = int(h * img_width / w)
    new_height = 67
    img_size = (img_width, new_height)
    resized = cv.resize(img, img_size)    # Convert to grayscale
    # gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    # img = Utils.skin_color_thresholding(img)
    img = cv.normalize(resized, None, 0, 255,
                                       cv.NORM_MINMAX).astype('uint8')
    # img = Utils.getThresholdedHand(img)
    # img = Utils.getMaskedHand(img) 
    # kp, descriptor = sift.detectAndCompute(img, None)
    # print(f"descriptor shape {descriptor.shape}")
    # img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
    # t_lower = 0  # Lower Threshold
    # t_upper = 255  # Upper threshold
    # edge = cv.Canny(img, t_lower, t_upper)

    # kernel = np.ones((30,30),np.uint8)
    # # img = cv.erode(img,kernel,iterations = 1)
    # img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    # img = cv.GaussianBlur(img, (7, 7), 0)

    # kp, descriptor = sift.detectAndCompute(img, None)
    # # print(f"descriptor shape {descriptor.shape}")
    # img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)


    outPath = os.path.join(outputPath, f'({i})1.JPG')
    cv.imwrite(outPath, img)
    # if cv.waitKey(1) & 0xff == 27:
    #     break
# imgPath = os.path.join(testImgPath, f'Screenshot 2023-05-08 035552.png')
# img = cv.imread(imgPath)
# img = Utils.getGuassianThresholdedHand(img)
# outPath = os.path.join(outputPath, f'test.png')
# cv.imwrite(outPath, img)