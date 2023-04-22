import os
import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn import cluster
import pickle
menPath = "../dataset_sample/men/"
womenPath = "../dataset_sample/Women/"
# menPath = "../Dataset_0-5/men/"
# womenPath = "../Dataset_0-5/Women/"
img = cv.imread(menPath + '0/0_men (1).jpg')
sift = cv.SIFT_create()
kp, descriptor = sift.detectAndCompute(img, None)
feature_set = np.copy(descriptor)
descriptors = []
descriptors.append(descriptor)
bagOfWords = []
y = []
y.append(0)
# for g in range(0, 11):
#     if g == 0:
#         c = 2
#     else:
#         c = 1
#     for i in range(c, 1251):
#         # Read image
#         img = cv.imread(path + f'{g}/{g}_men ({i}).jpg')
#         # Initialize sift
#         sift = cv.SIFT_create()
#         # Keypoints, descriptors
#         kp, descriptor = sift.detectAndCompute(img, None)
#         # Each keypoint has a descriptor with length 128
#         print(f"SIFT {g}/{i}")
#         if descriptor is None:
#             continue
#         else:
#             descriptors.append(np.array(descriptor))
#             feature_set = np.concatenate((feature_set, descriptor), axis=0)
#             y.append(g)


def read_images_from_folders(base_dir):
    global feature_set, y
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                (file_base_name, file_extension) = os.path.splitext(file_path)
                if os.path.isfile(file_path) and file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"SIFT {class_name} --- {file_base_name}")
                    # Read image
                    img = cv.imread(file_path)
                    # img = cv.normalize(img, None, 0, 255,
                    #                    cv.NORM_MINMAX).astype('uint8')
                    # Initialize sift
                    sift = cv.SIFT_create()
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


print("Reading men images...")
read_images_from_folders(menPath)
print(f"Success")
print("Reading women images...")
read_images_from_folders(womenPath)
print(f"Success")
# Kmeans clustering on all training set
print(f"Running kmeans...")
n_clusters = 1600
np.random.seed(0)
k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
k_means.fit(feature_set)
# Produce "bag of words" histogram for each image
print(f"Success")
print(f"Generating bag of words...")
for descriptor in descriptors:
    vq = [0] * n_clusters
    descriptor = k_means.predict(descriptor)
    for feature in descriptor:
        vq[feature] = vq[feature] + 1
    bagOfWords.append(vq)

print(f"Success")
print(f"Training SVM model...")
# Train the SVM multiclass classification model
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(bagOfWords, y)
print(f"Success")
print(f"Saving models")

# save the kmeans model to disk
filename1 = 'kmeans_model.sav'
pickle.dump(k_means, open(filename1, 'wb'))
# save the SVM model to disk
filename2 = 'gestures_model.sav'
pickle.dump(clf, open(filename2, 'wb'))
print(f"Success")
