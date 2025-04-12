from skimage.feature import hog
import numpy as np

def extract(X_train, X_test):
    def hog_features(images):
        return np.array([hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False) for img in images])
    return hog_features(X_train), hog_features(X_test)
