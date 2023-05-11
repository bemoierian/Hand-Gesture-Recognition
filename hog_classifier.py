import os
import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn import cluster
import pickle
from utils import Utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from skimage.feature import hog
# menPath = "../dataset_sample/men/"
# womenPath = "../dataset_sample/Women/"
menPath = "../Dataset_0-5/men/"
womenPath = "../Dataset_0-5/Women/"
trainingImgs = []
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
                if os.path.isfile(file_path) and file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"Reading {class_name} --- {file_name}")
                    # Read image
                    # img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    img = cv.imread(file_path)
                    # Calculate new size
                    # h, w = img.shape[:2]
                    # new_height = int(h * img_width / w)
                    # img_size = (img_width, new_height)
                    # resized = cv.resize(img, img_size)
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    # NormalizedImg = cv.normalize(gray, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
                    trainingImgs.append(gray)
                    y.append(class_name)

                    # img = cv.normalize(img, None, 0, 255,
                    #                    cv.NORM_MINMAX).astype('uint8')
                   
                    # fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    #     cells_per_block=(1, 1), visualize=True, channel_axis=-1)
                    # fd = hog.compute(img)
                    # if fd is None:
                    #     continue
                    # else:
                    #     hogFeatures.append(np.array(fd))
                    #     y.append(class_name)
                i = i + 1
                if i > 70:
                    break


print("Reading men images...")
read_images_from_folders(menPath)
print(f"Success")
print("Reading women images...")
read_images_from_folders(womenPath)
print(f"Success")

# Compute HOG features
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
    # print(f"HOG {trainingImgs.index(img)}")
    # hog from skimage
    # features, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
    #                     cells_per_block=(1, 1), visualize=True)
    # hog from opencv
    features = hog.compute(img)
    hogFeatures.append(features)

# ----------------------Train SVM---------------------
# print(f"Training SVM model...")
# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(hogFeatures, y)

# ----------------------Train AdaBoost---------------------
print(f"Training AdaBoost model...")
clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
clf.fit(hogFeatures, y)

print(f"Success")


print(f"Saving models")
# save the model to disk
filename2 = 'Hog_model.sav'
pickle.dump(clf, open(filename2, 'wb'))
print(f"Success")
