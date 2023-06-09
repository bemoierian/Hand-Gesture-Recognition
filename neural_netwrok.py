import os
import numpy as np
import cv2 as cv
from sklearn import svm
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd

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
menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"
# menPath = "../resized/men/"
# womenPath = "../resized/Women/"
inputImgs = []
hogFeatures = []
y = []
# hog = cv.HOGDescriptor()
# Set desired image width
img_width = 256


def read_images_from_folders(base_dir):
    global feature_set, y, img_width
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            i = 0
            for file_name in sorted(os.listdir(class_dir), key=Utils.extractInteger):
                file_path = os.path.join(class_dir, file_name)
                (file_base_name, file_extension) = os.path.splitext(file_path)
                if os.path.isfile(file_path) and file_extension.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                ]:
                    print(f"Reading {class_name} --- {file_name}")
                    # ------------------Read image---------------
                    img = cv.imread(file_path)
                    # ------------------Preprocessing---------------
                    img = Utils.getMaskedHand(img)
                    # Calculate new size
                    h, w = img.shape[:2]
                    new_height = int(h * img_width / w)
                    img_size = (img_width, new_height)
                    resized = cv.resize(img, img_size)
                    # gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
                    NormalizedImg = cv.normalize(
                        resized, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX
                    )
                    inputImgs.append(NormalizedImg)
                    y.append(int(class_name))
                # i = i + 1
                # if i > 20:
                #     break


# -----------------------READ IMAGES----------------
print("Reading men images...")
read_images_from_folders(menPath)
print(f"Success")
print("Reading women images...")
read_images_from_folders(womenPath)
print(f"Success")
# ----------------SPLIT TRAINING AND TEST SET----------------
# best random_state++ till now: 74
trainingImgs, testImgs, y_train, y_test = train_test_split(
    inputImgs, y, test_size=0.2, random_state=74
)
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
# hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
hog = cv.HOGDescriptor()
for img in trainingImgs:
    # hog from opencv
    features = hog.compute(img)
    hogFeatures.append(features)

# ----------------------PCA---------------------
print("PCA...")
pca = PCA(n_components=0.8)
pcaModel = pca.fit(hogFeatures)
print(f"Success")
print(f"Saving pca model")
# save the model to disk
filename1 = "pca.sav"
pickle.dump(pcaModel, open(filename1, "wb"))
hogFeatures = pcaModel.transform(hogFeatures)

print(f"Success")


# # ----------------------Train SVM---------------------
# print(f"Training SVM model...")
# clf = svm.SVC(decision_function_shape="ovo")
# clf.fit(hogFeatures, y_train)

# # ----------------------Train AdaBoost---------------------
# # print(f"Training AdaBoost model...")
# # clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
# # clf.fit(hogFeatures, y_train)

# print(f"Success")


# print(f"Saving models")
# # save the model to disk
# filename2 = "Hog_model.sav"
# pickle.dump(clf, open(filename2, "wb"))
# print(f"Success")
# # Load SVM model
# # clf = Utils.loadSVMModel()
# # ----------------------Test---------------------
print(f"Testing...")
hogFeaturesTest = []
# y_predict = []
# outputPath = "../predicted_images/"

for img in testImgs:
    # --------------hog-----------------
    features = hog.compute(img)
    hogFeaturesTest.append(features)

#     # ------------Predict---------------
#     # predictedClass = int(clf.predict([features])[0])
#     # y_predict.append(predictedClass)
#     # print(f"Predicted: {predictedClass}, True: {y_test[i]}")

#     # # -----------Draw predicted class on image and save it----------
#     # cv.putText(testImgs[i], f'{predictedClass}', (40, 80),
#     #                 cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 6)
#     # outPath = os.path.join(
#     #     outputPath, f'{y_test[i]}_{predictedClass} ({i}).JPG')
#     # cv.imwrite(outPath, testImgs[i])
hogFeaturesTest = pcaModel.transform(hogFeaturesTest)
# y_predict = clf.predict(hogFeaturesTest)

# accuracy = accuracy_score(y_test, y_predict)
# print(f"Accuracy: {accuracy}")


# ----------------Train Neural Network----------------
print(hogFeatures.shape)
print(len(y_train))
print(hogFeaturesTest.shape)
print(len(y_test))


# Build Neural Network model
model = keras.Sequential(
    [
        keras.layers.Dense(
            hogFeatures.shape[0],
            activation="relu",
            input_shape=(None, hogFeatures.shape[1]),
        ),
        keras.layers.Dense(6, activation="softmax"),
    ]
)

model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

train_x = np.array(hogFeatures).tolist()
train_y = np.array(y_train).tolist()
test_x = np.array(hogFeaturesTest).tolist()
test_y = np.array(y_test).tolist()

# Train model
model.fit(train_x, y_train, epochs=10, batch_size=128)

# Evaluate model
test_loss, test_acc = model.evaluate(test_x, test_y)
print("Test accuracy:", test_acc)

# Save model
# tf.saved_model.save(model, "my_model")
