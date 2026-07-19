"""
Nhanh 1: HOG features + Linear SVM (co dien, khong can GPU).
"""
import time

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


def extract_hog_batch(images: np.ndarray) -> np.ndarray:
    """Extract HOG descriptors for a batch of RGB images."""
    feats = []
    for img in images:
        gray = rgb2gray(img)
        f = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        feats.append(f)
    return np.array(feats)


def train_and_eval_svm(X_train_img, y_train_idx, X_test_img, y_test_idx):
    """Extract HOG features, train a LinearSVC, and report accuracy / macro-F1."""
    t0 = time.time()
    X_train_hog = extract_hog_batch(X_train_img)
    X_test_hog = extract_hog_batch(X_test_img)
    print(f"HOG feature dim: {X_train_hog.shape[1]} | extract time: {time.time() - t0:.1f}s")

    svm = LinearSVC(C=1.0, max_iter=5000)
    svm.fit(X_train_hog, y_train_idx)

    pred_svm = svm.predict(X_test_hog)
    acc_svm = accuracy_score(y_test_idx, pred_svm)
    f1_svm = f1_score(y_test_idx, pred_svm, average="macro")
    print(f"[SVM] accuracy = {acc_svm:.4f} | macro-F1 = {f1_svm:.4f}")

    return svm, acc_svm, f1_svm
