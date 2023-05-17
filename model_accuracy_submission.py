import os
import numpy as np
import cv2 as cv
# from sklearn import svm
from utils import Utils
# from sklearn.decomposition import PCA
import time
import pickle
inputPath = "./data"
inputImgs = []
y_predict = []
timeList = []

img_width = 128
def read_images_from_folders(base_dir):
    global inputImgs, img_width
    for file_name in sorted(os.listdir(base_dir), key=Utils.extractInteger):
        file_path = os.path.join(base_dir, file_name)
        if os.path.isfile(file_path):
            print(f"Reading {file_name}")
            # ------------------Read image-----------------
            img = cv.imread(file_path)
            # ----------------Append to list---------------
            inputImgs.append(img)



# ----------------------Load PCA---------------------
pcaModel = Utils.loadPCAModel()
print(f"Success")
# ----------------------Load SVM---------------------
clf = Utils.loadSVMModel()
print(f"Success")
# -----------------------READ IMAGES----------------
print("Reading input images...")
read_images_from_folders(inputPath)
print(f"Success")
# -------------------------HOG----------------------------
print("Preprocessing - HOG - PCA - Prediction...")
# Set HOG parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
# Create HOG descriptor
hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
for img in inputImgs:
    start = time.time()
    # ------------------Preprocessing---------------
    # Mask background and leave the hand in greyscale
    img = Utils.extract_hand(img,False, img_width)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ----------------------hog-----------------------
    features_hog = hog.compute(gray)
    feature_lbp = Utils.get_9ULBP(gray)
    features = np.concatenate((features_hog, feature_lbp), axis=None)
    # ----------------------PCA-----------------------
    features =  pcaModel.transform([features])
    # -------------------SVM Predict------------------
    predictedClass = int(clf.predict(features)[0])
    end = time.time()
    timeTaken = end - start
    timeList.append(timeTaken)
    y_predict.append(predictedClass)

# ----------------Save output to files--------------
print("Saving output to files...")
f = open("results.txt", "w")
for i in range(len(y_predict)):
    f.write(f"{y_predict[i]}\n")
f.close()

f = open("time.txt", "w")
for i in range(len(timeList)):
    timeList[i] = round(timeList[i], 3)
    f.write(f"{timeList[i]}\n")
f.close()

