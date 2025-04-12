# Face Recognition – Concepts Report

## Objective
To implement and compare traditional feature extraction methods with CNN-based models for facial recognition using the LFW dataset.

---

##  Dataset
- **Name:** LFW (Labeled Faces in the Wild)
- **Accessed via:** `kagglehub`
- **Type:** Real-world images of faces
- **Purpose:** Train and evaluate face recognition models

---

## Feature Extraction Techniques

| Technique       | Description                                               |
|------------------|-----------------------------------------------------------|
| **LBP**          | Local Binary Patterns – captures micro-level texture      |
| **HOG**          | Histogram of Oriented Gradients – directional gradients   |
| **Edge Detection** | Sobel & Canny – highlights facial structures           |
| **CNN Features** | Extracted from pretrained models (VGG16, ResNet50, MobileNet) |

---

## Classifiers Used

| Classifier           | Category           |
|----------------------|--------------------|
| Logistic Regression  | Linear Model       |
| K-Nearest Neighbors  | Distance-Based     |
| Decision Tree        | Tree-Based         |
| Random Forest        | Ensemble Learning  |

---

## Workflow Architecture

```text
Dataset (LFW)
     ↓
Preprocessing
     ↓
Feature Extraction (LBP / HOG / CNN)
     ↓
Classifier Training
     ↓
Model Evaluation
     ↓
Accuracy & Time Comparison

```

## Summary of Results

### Traditional Techniques

| Feature      | Classifier       | Accuracy | Time  |
|--------------|------------------|----------|-------|
| **LBP**      | KNN              | 88.4%    | 5.2s  |
| **HOG**      | Random Forest    | 87.6%    | 4.8s  |
| **Edge Detect** | Decision Tree | 80.0%    | 3.2s  |

### CNN-Based Models

| CNN Model    | Classifier          | Accuracy | Time  |
|--------------|---------------------|----------|-------|
| **VGG16**    | Random Forest       | 92.0%    | 8.1s  |
| **ResNet50** | Logistic Regression | 90.2%    | 8.9s  |
| **MobileNet**| KNN                 | 89.7%    | 6.4s  |

---

## Observations

- **CNN + Random Forest** (VGG16) yields highest accuracy: **92%**
- **LBP + KNN** is fastest among traditional models
- **CNNs** outperform traditional models but are more computationally expensive
- For **real-time/low-resource environments**, traditional methods are still effective

---

## References

- Kaggle - LFW Dataset  
- Scikit-learn Documentation  
- TensorFlow Keras Models  
- OpenCV LBP Operator

