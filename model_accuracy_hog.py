import cv2 as cv
from sklearn.metrics import accuracy_score
import os
import numpy as np
from utils import Utils
from sklearn.decomposition import PCA
import time
# from skimage.feature import hog

# Load SVM model
clf = Utils.loadSVMModel()
print("Success")
y_true = []
y_predict = []
set1Path = "../Set1/"
set2Path = "../Set2/"
outputPath = "../predicted_images/"

# Compute HOG features
# Set HOG parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

# # Create HOG descriptor
hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

inputImgs = []
timeList = []

img_width = 128
def read_images_from_folders(base_dir):
    global inputImgs, y_true, img_width
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                (file_base_name, file_extension) = os.path.splitext(file_path)
                if os.path.isfile(file_path) and file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"Reading {class_name}/{file_name}")
                    # ------------------Read image---------------
                    img = cv.imread(file_path)
                    # ------------------Append to list---------------
                    inputImgs.append(img)
                    y_true.append(int(class_name))

# ----------------------Load PCA---------------------
pcaModel = Utils.loadPCAModel()
print(f"Success")
# ----------------------Load SVM---------------------
clf = Utils.loadSVMModel()
print(f"Success")
# -----------------------READ IMAGES----------------
print("Reading input images...")
read_images_from_folders(set1Path)
read_images_from_folders(set2Path)
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
for i in range(len(inputImgs)):
    img = inputImgs[i]
    start = time.time()
    # ------------------Preprocessing---------------
    #  Reduce highlights and increase shadows
    img = Utils.adjust_image(img)
    # Mask background and leave the hand in greyscale
    img = Utils.extract_hand(img,False, img_width)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = Utils.getMaskedHand(img)
    # # Calculate new size
    # h, w = img.shape[:2]
    # new_height = 72
    # img_size = (img_width, new_height)
    # resized = cv.resize(img, img_size)
    # ----------------------hog-----------------------
    features_hog = hog.compute(gray)
    # ----------------------LBP-----------------------
    # feature_lbp = Utils.get_9ULBP(gray)
    # features = np.concatenate((features_hog, feature_lbp), axis=None)
    # ----------------------PCA-----------------------
    features =  pcaModel.transform([features_hog])
    # -------------------SVM Predict------------------
    predictedClass = int(clf.predict(features)[0])
    end = time.time()
    timeTaken = end - start
    timeList.append(timeTaken)
    y_predict.append(predictedClass)
    print(f"Predicted: {predictedClass}, True: {y_true[i]}")
    # ---------Draw predicted class on image and save it-------------
    cv.putText(gray, f'{predictedClass}', (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 6)
    outPath = os.path.join(
        outputPath, f'{y_true[i]}_({i}).JPG')
    cv.imwrite(outPath, gray)


# ----------------Save output to files--------------
# print("Saving output to files...")
# f = open("results.txt", "w")
# for i in range(len(y_predict)):
#     f.write(f"{y_predict[i]}\n")
# f.close()

f = open("time.txt", "w")
for i in range(len(timeList)):
    timeList[i] = round(timeList[i], 3)
    f.write(f"{timeList[i]}\n")
f.close()

accuracy = accuracy_score(y_true, y_predict)
print(f"Accuracy: {accuracy}")
