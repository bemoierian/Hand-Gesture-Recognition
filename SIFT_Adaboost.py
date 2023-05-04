import math
import os
import pickle
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utils import Utils

def train_adaboost_sift(training_images, training_labels, T, weak_classifier):
    N = len(training_images)
    Np = sum(training_labels)
    Nn = N - Np
    w = np.zeros((T+1, N))
    w[0] = np.where(training_labels == 0, 1/(2*Nn), 1/(2*Np))
    
    for m in range(T):
        w[m] /= np.sum(w[m])
        fm, pm, tm = weak_classifier(w[m], training_images, training_labels)
        hm = lambda I: h(I, fm, pm, tm)
        e = np.array([w[m,i]*np.abs(hm(I)-l) for i,(I,l) in enumerate(zip(training_images, training_labels))])
        em = np.sum(e)
        beta_m = em / (1-em)
        w[m+1] = w[m] * np.power(beta_m, 1-e)
    
    alpha = np.log(1/beta_m)
    H = lambda x: int(np.sum(alpha * h(x,fm,pm,tm)) > T)
    return H

def h(I, fm, pm, tm):
    global k_means
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(I,None)
    des = k_means.predict(des)
    return pm * (des[fm] - tm)

def weak_classifier(wm, training_images, training_labels):
    global k_means
    sift = cv2.SIFT_create()
    des_list = []
    for I in training_images:
        kp, des = sift.detectAndCompute(I,None)
        if des is None:
            des = np.zeros((1,128)) + 1e-5
        des = k_means.predict(des)
        des_list.append(des)
    des = np.vstack(des_list)
    
    fm = 0
    min_em = float('inf')
    pm = 1
    tm = 0
    
    for f in range(des.shape[1]):
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(des[:,f].reshape(-1,1), training_labels, sample_weight=wm)
        thres = tree.tree_.threshold[0]
        left = tree.tree_.children_left[0]
        right = tree.tree_.children_right[0]
        p = 1 if tree.tree_.value[left][0][1] > tree.tree_.value[right][0][1] else -1
        kp, descrip = sift.detectAndCompute(I,None)
        descrip = k_means.predict(descrip)

        hm = lambda I: p * (descrip[:,f] - thres)
        e = np.array([wm[i]*np.abs(hm(I)-l) for i,(I,l) in enumerate(zip(training_images, training_labels))])
        em = np.sum(e)
        if em < min_em:
            fm = f
            min_em = em
            pm = p
            tm = thres
    
    return fm, pm, tm

def read_images_from_folders(base_dir):
    global training_images, training_labels
    # img_shape = (400, 400)

    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            i = 0
            for file_name in sorted(os.listdir(class_dir), key=Utils.extractInteger):
                file_path = os.path.join(class_dir, file_name)
                (file_base_name, file_extension) = os.path.splitext(file_path)
                if os.path.isfile(file_path) and file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    # img = cv2.resize(img, img_shape)
                    training_images.append(img)
                    training_labels.append(int(class_name))
    training_labels = np.array(training_labels)
menPath = "../dataset_sample/men/"
womenPath = "../Dataset_0-5/Women/"
# Load kmeans model
k_means = Utils.loadKmeansModel()
print("Success")
# Load training images and labels
training_images = []
training_labels = []
read_images_from_folders(menPath)
print(len(training_images))
print(len(training_labels))
# Train Adaboost classifier
T = 100
H = train_adaboost_sift(training_images, training_labels, T)

# Test classifier on new image
# test_img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
# gesture = H(test_img) + 1
# print(f'Detected gesture: {gesture}')



# save the kmeans model to disk
filename1 = 'adaboost_model.sav'
pickle.dump(H, open(filename1, 'wb'))