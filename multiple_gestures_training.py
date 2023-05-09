import os
import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn import cluster
import pickle
from utils import Utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
# menPath = "../dataset_sample/men/"
# womenPath = "../dataset_sample/Women/"
menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"
img = cv.imread(menPath + '0/0_men (1).jpg')
sift = cv.SIFT_create()
kp, descriptor = sift.detectAndCompute(img, None)
feature_set = np.copy(descriptor)
descriptors = []
descriptors.append(descriptor)
bagOfWords = []
y = []
y.append(0)


def read_images_from_folders(base_dir):
    global feature_set, y, sift
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            i = 0
            for file_name in sorted(os.listdir(class_dir), key=Utils.extractInteger):
                file_path = os.path.join(class_dir, file_name)
                (file_base_name, file_extension) = os.path.splitext(file_path)
                if os.path.isfile(file_path) and file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"SIFT {class_name} --- {file_name}")
                    # Read image
                    # img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    img = cv.imread(file_path)

                    img = Utils.skin_color_thresholding(img)
                    img = cv.normalize(img, None, 0, 255,
                                       cv.NORM_MINMAX).astype('uint8')
                    kernel = np.ones((30,30),np.uint8)
                    # img = cv.erode(img,kernel,iterations = 1)
                    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
                    # img = Utils.getMaskedHand(img)
                    # img = Utils.getThresholdedHand(img)
                    # img = cv.normalize(img, None, 0, 255,
                    #                    cv.NORM_MINMAX).astype('uint8')
                    # Initialize sift
                    # sift = cv.SIFT_create()

                    # Keypoints, descriptors
                    kp, descriptor = sift.detectAndCompute(img, None)
                    # Each keypoint has a descriptor with length 128
                    if descriptor is None:
                        continue
                    else:
                        descriptors.append(np.array(descriptor))
                        feature_set = np.concatenate(
                            (feature_set, descriptor), axis=0)
                        y.append(class_name)
                i = i + 1
                if i > 70:
                    break


# print("Reading men images...")
# read_images_from_folders(menPath)
# print(f"Success")
# print("Reading women images...")
# read_images_from_folders(womenPath)
# print(f"Success")

# -------------------Save feature set and descriptors-------------------
# print("Saving feature set and descriptors...")
# np.save('feature_set.npy', feature_set)
# # descriptors = np.array(descriptors)
# np.save('descriptors.npy', descriptors)
# print(f"Success")

# pca = PCA(n_components=1600)
# pca.fit(feature_set)
# -------------------Load feature set and descriptors-------------------
# print("Loading feature set and descriptors...")
# feature_set = np.load('feature_set.npy', allow_pickle = True)
# descriptors = np.load('descriptors.npy', allow_pickle = True)

# -----------------------Train Kmeans------------------
print(f"Running kmeans...")
n_clusters = 10000
np.random.seed(0)
# k_means = cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=1)
# k_means.fit(feature_set)

# -----------------save the kmeans model to disk------------------
# filename1 = 'kmeans_model.sav'
# pickle.dump(k_means, open(filename1, 'wb'))

# -----------------------Load saved Kmeans------------------
k_means = Utils.loadKmeansModel()
print(f"Success")
# -----------Produce "bag of words" histogram for each image----------
# print(f"Generating bag of words...")
# for descriptor in descriptors:
#     descriptor = k_means.predict(descriptor)
#     vq = [0] * n_clusters
#     for feature in descriptor:
#         vq[feature] = vq[feature] + 1
#     bagOfWords.append(vq)
# print(f"Success")

# -------------------Save bag of words and y-------------------
# print("Saving bag of words and y...")
# bagOfWords = np.array(bagOfWords)
# np.save('bag_of_words.npy', bagOfWords)
# y = np.array(y)
# np.save('y.npy', y)
# print(f"Success")

# -------------------Load bag of words and y-------------------
print("Loading bag of words and y...")
bagOfWords = np.load('bag_of_words.npy', allow_pickle = True)
y = np.load('y.npy', allow_pickle = True)

# ----------------------Train SVM---------------------
# print(f"Training SVM model...")
# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(bagOfWords, y)

# ----------------------Train AdaBoost---------------------
print(f"Training AdaBoost model...")
clf = AdaBoostClassifier(n_estimators=50000, random_state=0)
clf.fit(bagOfWords, y)

print(f"Success")


print(f"Saving models")
# save the model to disk
filename2 = 'gestures_model.sav'
pickle.dump(clf, open(filename2, 'wb'))
print(f"Success")
