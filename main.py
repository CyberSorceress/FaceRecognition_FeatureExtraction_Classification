from models import hog_features, lbp_features, edge_features, cnn_features, classifiers
from utils import preprocessing, evaluation, plot_utils
import os

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocessing.load_data("data/lfw_dataset")

# Feature extraction
features = {
    "HOG": hog_features.extract(X_train, X_test),
    "LBP": lbp_features.extract(X_train, X_test),
    "Edge": edge_features.extract(X_train, X_test),
    "VGG16": cnn_features.extract(X_train, X_test, model="VGG16"),
    "ResNet50": cnn_features.extract(X_train, X_test, model="ResNet50"),
    "MobileNet": cnn_features.extract(X_train, X_test, model="MobileNet")
}

# Classifiers
classifiers_to_run = ["LogisticRegression", "KNN", "DecisionTree", "RandomForest"]

# Run all models
for method, (f_train, f_test) in features.items():
    print(f"\n===== Using {method} Features =====")
    for clf in classifiers_to_run:
        model = classifiers.train_model(clf, f_train, y_train)
        y_pred = model.predict(f_test)
        print(f"\n--- {clf} Classifier ---")
        evaluation.print_scores(y_test, y_pred)
        plot_utils.plot_confusion(y_test, y_pred, title=f"{method}_{clf}_Confusion")
        plot_utils.plot_roc(model, f_test, y_test, title=f"{method}_{clf}_ROC")
