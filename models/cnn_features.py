from tensorflow.keras.applications import VGG16, ResNet50, MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

def extract(X_train, X_test, model="VGG16"):
    base_model = {
        "VGG16": VGG16,
        "ResNet50": ResNet50,
        "MobileNet": MobileNet
    }[model](weights="imagenet", include_top=False, input_shape=(128, 128, 3))

    model = Model(inputs=base_model.input, outputs=base_model.output)
    def get_features(data):
        data_rgb = np.repeat(data[..., np.newaxis], 3, -1)  # grayscale to RGB
        data_pre = preprocess_input(data_rgb)
        features = model.predict(data_pre)
        return features.reshape((features.shape[0], -1))
    
    return get_features(X_train), get_features(X_test)
