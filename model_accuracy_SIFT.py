import cv2 as cv
from sklearn.metrics import accuracy_score
import os
import numpy as np
from utils import Utils
# Load kmeans model
k_means = Utils.loadKmeansModel()
print("Success")
n_clusters = 10000
# Load SVM model
clf = Utils.loadSVMModel()
print("Success")
y_true = []
y_predict = []
menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"
outputPath = "../predicted_images/"
bagOfWords = []


def processImages(imgsPath, classeStartIndex, classeEndIndex, imgStartIndex, imgEndIndex):
    global menPath, womenPath, y_true, y_predict, k_means, clf, outputPath, n_clusters, bagOfWords
    sift = cv.SIFT_create()
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
            # img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
            # img = Utils.getMaskedHand(img)

            # img = Utils.getThresholdedHand(img)


            img = Utils.skin_color_thresholding(img)
            img = cv.normalize(img, None, 0, 255,
                                cv.NORM_MINMAX).astype('uint8')
            kernel = np.ones((30,30),np.uint8)
            # img = cv.erode(img,kernel,iterations = 1)
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

            # Feature extraction
            kp, descriptor = sift.detectAndCompute(img, None)
            # Produce "bag of words" vector
            descriptor = k_means.predict(descriptor)
            print(f"SIFT {g}/{i}")
            vq = [0] * n_clusters
            for feature in descriptor:
                vq[feature] = vq[feature] + 1  # load the model from disk
            bagOfWords.append(vq)
            # append true class
            y_true.append(g)

            # Predict the result
            predictedClass = int(clf.predict([vq])[0])
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
# accuracy =clf.score(bagOfWords, y_true)
# print(f"Accuracy: {accuracy}")

# print(f"y_true: {y_true}")
# print(f"y_predict: {y_predict}")
accuracy = accuracy_score(y_true, y_predict)
print(f"Accuracy: {accuracy}")
