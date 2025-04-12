from skimage.feature import local_binary_pattern
import numpy as np

def extract(X_train, X_test):
    def lbp_hist(img):
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        return hist / hist.sum()
    return np.array([lbp_hist(i) for i in X_train]), np.array([lbp_hist(i) for i in X_test])
