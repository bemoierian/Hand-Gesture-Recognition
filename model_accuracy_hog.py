import cv2 as cv
from sklearn.metrics import accuracy_score
import os
import numpy as np
from utils import Utils
from skimage.feature import hog

# Load SVM model
clf = Utils.loadSVMModel()
print("Success")
y_true = []
y_predict = []
menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"
outputPath = "../predicted_images/"
testImgs = []

# Compute HOG features
# Set HOG parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

# # Create HOG descriptor
# hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
hog = cv.HOGDescriptor()
img_width = 256


def processImages(imgsPath, classeStartIndex, classeEndIndex, imgStartIndex, imgEndIndex):
    global menPath, womenPath, y_true, y_predict, clf, outputPath, hog, testImgs
    for g in range(classeStartIndex, classeEndIndex):
        if imgsPath == menPath:
            print(f"Men {g}")
        else:
            print(f"Women {g}")
        class_dir = os.path.join(imgsPath, f"{g}")
        for i in range(imgStartIndex, imgEndIndex):
            # Read image
            imgPath = os.path.join(
                class_dir, f'{g}{"_men" if imgsPath == menPath else "_woman"} ({i}).JPG')
            img = cv.imread(imgPath)
            # Calculate new size
            h, w = img.shape[:2]
            new_height = int(h * img_width / w)
            img_size = (img_width, new_height)
            resized = cv.resize(img, img_size)
            gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
            NormalizedImg = cv.normalize(gray, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            # testImgs.append(NormalizedImg)
            # fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
            #             cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            features = hog.compute(NormalizedImg)
            # features, hog_image = hog(NormalizedImg, orientations=8, pixels_per_cell=(16, 16),
            #             cells_per_block=(1, 1), visualize=True)

            # img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
            # img = Utils.getMaskedHand(img)

            # img = Utils.getThresholdedHand(img)


            # img = cv.normalize(img, None, 0, 255,
            #                     cv.NORM_MINMAX).astype('uint8')
          
            # append true class
            y_true.append(g)

            # Predict the result
            predictedClass = int(clf.predict([features])[0])
            y_predict.append(predictedClass)
            print(f"Predicted class: {predictedClass}, True class: {g}")

            # # Draw predicted class on image and save it
            # cv.rectangle(img, (5, 5), (500, 100), (175, 0, 175), cv.FILLED)
            # cv.putText(img, f'{predictedClass}', (40, 80),
            #                 cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
            # outPath = os.path.join(
            #     outputPath, f'{g}{"_men" if imgsPath == menPath else "_woman"} ({i}).JPG')
            # cv.imwrite(outPath, img)


processImages(menPath, 0, 6, 71, 91)
processImages(womenPath, 0, 6, 71, 91)
# for img in testImgs:
#     # print(f"HOG {trainingImgs.index(img)}")
#     features = hog.compute(img)
#     # Predict the result
#     predictedClass = int(clf.predict([features])[0])
#     y_predict.append(predictedClass)
#     print(f"Predicted class: {predictedClass}, True class: {g}")
# accuracy =clf.score(bagOfWords, y_true)
# print(f"Accuracy: {accuracy}")

# print(f"y_true: {y_true}")
# print(f"y_predict: {y_predict}")
accuracy = accuracy_score(y_true, y_predict)
print(f"Accuracy: {accuracy}")
