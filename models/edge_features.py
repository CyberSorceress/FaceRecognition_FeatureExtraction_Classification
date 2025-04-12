from skimage.filters import sobel
import numpy as np

def extract(X_train, X_test):
    def sobel_feat(img):
        edge_img = sobel(img)
        return edge_img.flatten()
    return np.array([sobel_feat(i) for i in X_train]), np.array([sobel_feat(i) for i in X_test])
