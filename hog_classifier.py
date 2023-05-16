import os
import numpy as np
import cv2 as cv
from sklearn import svm
# from sklearn import cluster
import pickle
from utils import Utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
# from skimage.feature import hog

# menPath = "../dataset_sample/men/"
# womenPath = "../dataset_sample/Women/"
# menPath = "../Dataset_0-5/men/"
# womenPath = "../Dataset_0-5/Women/"
menPath = "../resized/men/"
womenPath = "../resized/Women/"
inputImgs = []
hogFeatures = []
y = []
# hog = cv.HOGDescriptor()
# Set desired image width
img_width = 120
def read_images_from_folders(base_dir):
    global inputImgs, y, img_width
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            i = 0
            for file_name in sorted(os.listdir(class_dir), key=Utils.extractInteger):
                file_path = os.path.join(class_dir, file_name)
                (file_base_name, file_extension) = os.path.splitext(file_path)
                if os.path.isfile(file_path) and file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"Reading {file_name}")
                    # ------------------Read image---------------
                    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    # ------------------Preprocessing---------------
                    # # Reduce highlights and increase shadows
                    # img = Utils.adjust_image(img)
                    # # Mask background and leave the hand in greyscale
                    # img = Utils.getMaskedHand(img)
                    # # Calculate new size
                    # h, w = img.shape[:2]
                    # new_height = int(h * img_width / w)
                    # img_size = (img_width, new_height)
                    # resized = cv.resize(img, img_size)
                    # NormalizedImg = cv.normalize(resized, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
                    # ------------------Append to list---------------
                    inputImgs.append(img)
                    y.append(int(class_name))
                # i = i + 1
                # if i > 40:
                #     break

# -----------------------READ IMAGES----------------
print("Reading men images...")
read_images_from_folders(menPath)
print(f"Success")
print("Reading women images...")
read_images_from_folders(womenPath)
print(f"Success")
# ----------------SPLIT TRAINING AND TEST SET----------------
# best random_state till now: 74, 693
trainingImgs, testImgs, y_train, y_test = train_test_split(inputImgs, y, test_size=0.2, random_state=693)
print(f"Training images: {len(trainingImgs)}")
print(f"Test images: {len(testImgs)}")
# -------------------------HOG----------------------------
# Set HOG parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
# Create HOG descriptor
hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
for img in trainingImgs:
    # -----------------hog------------------
    features = hog.compute(img)
    hogFeatures.append(features)

# ----------------------PCA---------------------
print("PCA...")
pca = PCA(n_components=0.83)
pcaModel = pca.fit(hogFeatures)
print(f"PCA components: {pcaModel.n_components_}")
print(f"Success")
print(f"Saving pca model")
# save the model to disk
filename1 = 'pca.sav'
pickle.dump(pcaModel, open(filename1, 'wb'))
hogFeatures = pcaModel.transform(hogFeatures)
print(f"size of hogFeatures: {hogFeatures.shape}")
print(f"Success")
# ----------------------Train SVM---------------------
print(f"Training SVM model...")
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(hogFeatures, y_train)

# ----------------------Train AdaBoost---------------------
# print(f"Training AdaBoost model...")
# clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
# clf.fit(hogFeatures, y_train)

print(f"Success")


print(f"Saving models")
# save the model to disk
filename2 = 'SVM_model.sav'
pickle.dump(clf, open(filename2, 'wb'))
print(f"Success")
# Load SVM model
# clf = Utils.loadSVMModel()
# ----------------------Test---------------------
print(f"Testing...")
hogFeaturesTest = []
y_predict = []
outputPath = "../predicted_images/"

for img in testImgs:
    # --------------hog-----------------
    features = hog.compute(img)
    hogFeaturesTest.append(features)

    # ------------Predict---------------
    # predictedClass = int(clf.predict([features])[0])
    # y_predict.append(predictedClass)
    # print(f"Predicted: {predictedClass}, True: {y_test[i]}")

    # # -----------Draw predicted class on image and save it----------
    # cv.putText(testImgs[i], f'{predictedClass}', (40, 80),
    #                 cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 6)
    # outPath = os.path.join(
    #     outputPath, f'{y_test[i]}_{predictedClass} ({i}).JPG')
    # cv.imwrite(outPath, testImgs[i])
hogFeaturesTest =  pcaModel.transform(hogFeaturesTest)
y_predict = clf.predict(hogFeaturesTest)

accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy}")
