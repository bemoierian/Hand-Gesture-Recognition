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
menPathOutput = "../resized/men/"
womenPathOutput = "../resized/Women/"
img_width = 120
def read_images_from_folders(base_dir, output_dir):
    global feature_set, y, img_width
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        class_dir_output = os.path.join(output_dir, class_name)
        if os.path.isdir(class_dir):
            i = 0
            for file_name in sorted(os.listdir(class_dir), key=Utils.extractInteger):
                file_path = os.path.join(class_dir, file_name)
                file_path_output = os.path.join(class_dir_output, file_name)
                (file_base_name, file_extension) = os.path.splitext(file_path)
                if os.path.isfile(file_path) and file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"Reading {class_name} --- {file_name}")
                    # Read image
                    # img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    img = cv.imread(file_path)
                    img = Utils.gamma_correction(img, 0.7)
                    img = Utils.adjust_image(img)
                    img = Utils.getMaskedHand(img)
                    # Calculate new size
                    h, w = img.shape[:2]
                    new_height = int(h * img_width / w)
                    img_size = (img_width, new_height)
                    resized = cv.resize(img, img_size)
                    # gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
                    NormalizedImg = cv.normalize(resized, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
                    cv.imwrite(file_path_output, NormalizedImg)
                # i = i + 1
                # if i > 70:
                #     break


print("Reading men images...")
read_images_from_folders(menPath, menPathOutput)
print(f"Success")
print("Reading women images...")
read_images_from_folders(womenPath, womenPathOutput)
