from sklearn.model_selection import train_test_split
from skimage.io import imread_collection
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import os

def load_data(folder_path):
    images = []
    labels = []
    for label_folder in os.listdir(folder_path):
        full_path = os.path.join(folder_path, label_folder)
        for img_file in os.listdir(full_path):
            img = rgb2gray(resize(imread_collection([os.path.join(full_path, img_file)])[0], (128, 128)))
            images.append(img)
            labels.append(label_folder)
    X = np.array(images)
    y = np.array(labels)
    return train_test_split(X, y, test_size=0.2, random_state=42)
