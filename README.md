# Face Recognition using Feature Extraction & Classification

This project compares traditional and deep learning-based feature extraction methods (HOG, LBP, Edge, VGG16, ResNet50, MobileNet) for face recognition using classifiers like Logistic Regression, KNN, Decision Tree, and Random Forest.

### Dataset

- **Name**: Labeled Faces in the Wild (LFW)
- **Source**: [Kaggle – atulanandjha/lfwpeople](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)
- **How to Download**:
  The dataset is large and not stored in this repository directly.
  We use the `kagglehub` library to download it on-the-fly.

```bash
pip install kagglehub
pip install -r requirements.txt
python main.py
