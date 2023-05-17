import cv2 as cv
from sklearn.metrics import accuracy_score
import os
# import numpy as np
from utils import Utils
import time

import pickle
print("Success")
y_true = []
y_predict = []
menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"
outputPath = "../predicted_images/"
bagOfWords = []
inputImgs = []
timeList = []
set1Path = "../Set1/"
set2Path = "../Set2/"

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


# ----------------------Load SVM---------------------
filename2 = 'SIFT_SVM_model.sav'
clf = pickle.load(open(filename2, 'rb'))
print(f"Success")
# --------------------Load kmeans-------------------
k_means = Utils.loadKmeansModel()
print("Success")
n_clusters = 1600
# -----------------------READ IMAGES----------------
print("Reading input images...")
read_images_from_folders(set1Path)
read_images_from_folders(set2Path)
print(f"Success")
# -------------------------SIFT----------------------------
sift = cv.SIFT_create()
for i in range(len(inputImgs)):
    img = inputImgs[i]
    start = time.time()
    # ------------------Preprocessing---------------
    #  Reduce highlights and increase shadows
    img = Utils.adjust_image(img)
    # Mask background and leave the hand in greyscale
    img = Utils.extract_hand(img,False, img_width)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Feature extraction
    kp, descriptor = sift.detectAndCompute(gray, None)
    # Produce "bag of words" vector
    descriptor = k_means.predict(descriptor)
    vq = [0] * n_clusters
    for feature in descriptor:
        vq[feature] = vq[feature] + 1  # load the model from disk
    bagOfWords.append(vq)
    # --------------------Predict-------------------
    predictedClass = int(clf.predict([vq])[0])
    end = time.time()
    y_predict.append(predictedClass)
    print(f"Predicted: {predictedClass}, True: {y_true[i]}")
    timeTaken = end - start
    timeList.append(timeTaken)

    # Draw predicted class on image and save it
    cv.putText(img, f'{predictedClass}', (40, 80),
                    cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    outPath = os.path.join(
        outputPath, f'{y_true[i]}_({i}).JPG')
    cv.imwrite(outPath, img)
        
accuracy = accuracy_score(y_true, y_predict)
print(f"Accuracy: {accuracy}")
