# Skin Cancer Detection System
### A Deep Learning Approach for Melanoma Classification

**Medical Informatics Course — Term Project**

---

## Abstract

A deep learning-based system for detecting melanoma from dermoscopic images using transfer learning. The system employs **MobileNetV2**, a lightweight convolutional neural network architecture pre-trained on ImageNet, fine-tuned on the **HAM10000** dataset. The model achieves a test accuracy of **86.6%** and an AUC score of **82.7%** for binary classification between benign lesions and melanoma. A balanced data augmentation strategy addresses class imbalance using geometric transformations exclusively to preserve diagnostically critical color information. A web-based interface built with **Streamlit** is included for practical deployment.

**Keywords:** Deep Learning · Transfer Learning · Medical Image Analysis · Skin Cancer Detection · MobileNetV2 · Binary Classification

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Motivation

Skin cancer is one of the most common cancers worldwide. Melanoma, its most dangerous form, drops from a **99% to 27% 5-year survival rate** when detected late. Dermoscopic imaging is the clinical standard, yet accurate reading requires years of expertise. This project builds a computer-aided diagnosis (CAD) system to assist dermatologists with rapid, reproducible screening.

---

## Dataset

### HAM10000

The [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dataset contains **10,015 dermoscopic images** across seven diagnostic categories. For this project, labels are binarized:

| Class | Diagnoses Included | Count |
|-------|--------------------|-------|
| **Benign** | nv, bkl, bcc, akiec, vasc, df | ~9,000 (~90%) |
| **Melanoma** | mel | ~1,000 (~10%) |

### Preprocessing Pipeline

1. Resize to **224 × 224** pixels (MobileNetV2 input)
2. Convert BGR → RGB
3. Normalize pixel values to **[0, 1]**

### Data Splits (stratified)

| Split | Size |
|-------|------|
| Training | 70% (~7,000 images) |
| Validation | 15% (~1,500 images) |
| Test | 15% (~1,500 images) |

---

## Model Architecture

The backbone is **MobileNetV2** (ImageNet weights, top excluded). A custom classification head is appended:

```
MobileNetV2 backbone
    └── GlobalAveragePooling2D
        └── BatchNormalization
            └── Dense(256, ReLU)
                └── Dropout(0.5)
                    └── Dense(128, ReLU)
                        └── Dropout(0.3)
                            └── Dense(1, Sigmoid)   ← binary output
```

---

## Training Strategy

### Two-Phase Transfer Learning

| Phase | Epochs | Frozen Layers | Learning Rate |
|-------|--------|---------------|---------------|
| Feature Extraction | 30 | All MobileNetV2 layers | 0.001 |
| Fine-tuning | 10 | All except last 30 layers | 1e-5 |

### Data Augmentation

Color transformations (brightness, channel shift, contrast) are **intentionally excluded** to preserve diagnostically relevant color information. Only geometric transforms are applied:

| Transformation | Melanoma | Benign |
|----------------|----------|--------|
| Rotation | ±45° | ±20° |
| Zoom | 0.3 | 0.2 |
| Width/Height Shift | 0.3 | 0.2 |
| Shear | 15° | 10° |
| Horizontal/Vertical Flip | ✓ | ✓ |

A **balanced generator** equalizes melanoma and benign samples in each training batch to counter the 1:9 class imbalance.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Metrics | Accuracy, Precision, Recall, AUC |

### Callbacks

- **EarlyStopping** — monitors `val_loss`, patience = 5
- **ModelCheckpoint** — saves best model by `val_auc`
- **ReduceLROnPlateau** — halves LR when `val_loss` plateaus (patience = 3)

---

## Results

### Overall Metrics (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | **86.6%** |
| AUC | **82.7%** |
| Precision | 40.0% |
| Recall | 40.7% |
| Loss | 0.314 |

### Class-Level Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Benign | 96.4% | 74.7% | 84.2% |
| Melanoma | 27.8% | 77.8% | 40.9% |

### Confusion Matrix

|  | Predicted Benign | Predicted Melanoma |
|--|------------------|--------------------|
| **Actual Benign** | 998 (TN) | 338 (FP) |
| **Actual Melanoma** | 37 (FN) | 130 (TP) |

> The model prioritizes **melanoma recall (77.8%)** to minimize missed diagnoses — the most critical failure mode in a clinical screening context.

---

## Project Structure

```
MedicalInformatics/
├── codes/
│   ├── src/
│   │   ├── config.py          # Hyperparameters & paths
│   │   ├── data_loader.py     # Dataset loading & label conversion
│   │   ├── preprocessing.py   # Image preprocessing & augmentation
│   │   ├── model.py           # Model architecture
│   │   └── train.py           # Training pipeline
│   ├── models/                # Saved model & evaluation artifacts
│   │   ├── skin_cancer_model.h5
│   │   ├── model_metrics.json
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── training_history.png
│   ├── data/                  # Raw dataset (HAM10000)
│   ├── notebooks/             # Exploratory notebooks
│   ├── app.py                 # Streamlit web application
│   ├── requirements.txt       # Python dependencies
│   └── PROJECT_REPORT.md      # Full project report
└── README.md
```

---

## Setup & Usage

### 1. Install Dependencies

```bash
cd codes
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place it under `codes/data/raw/`.

### 3. Train the Model

```bash
cd codes
python src/train.py
```

### 4. Launch the Web Application

```bash
cd codes
streamlit run app.py
```

The Streamlit interface allows you to upload a dermoscopic image and receive an instant classification result with confidence score and model performance statistics.

---

## Limitations & Future Work

### Current Limitations

- **Low melanoma precision (27.8%)** — high false-positive rate increases unnecessary referrals
- Single dataset; may not generalize across skin types, imaging devices, or clinical photography
- Binary classification only; real-world diagnosis involves differential diagnosis across multiple lesion types
- MobileNetV2 trades maximum accuracy for efficiency

### Future Directions

1. **Multi-class classification** across all seven HAM10000 categories
2. **Ensemble methods** for improved robustness
3. **Larger architectures** — EfficientNet, Vision Transformers
4. **Explainability** — Grad-CAM or attention maps to visualize decisions
5. **External validation** on diverse populations and devices
6. **Active learning** to improve performance with limited labeled data
7. **Clinical API** for integration with hospital information systems

---

## Software & Libraries

| Library | Version |
|---------|---------|
| Python | 3.8+ |
| TensorFlow/Keras | 2.10+ |
| Streamlit | 1.28+ |
| OpenCV | 4.7+ |
| scikit-learn | 1.2+ |
| NumPy | 1.23+ |
| Pandas | 1.5+ |

---

## References

1. American Cancer Society. (2023). *Cancer Facts & Figures 2023*.
2. Esteva, A. et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542, 115–118.
3. Sandler, M. et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR 2018*, 4510–4520.
4. Johnson, J. M., & Khoshgoftaar, T. M. (2019). Survey on deep learning with class imbalance. *Journal of Big Data*, 6(1).
5. Tschandl, P. et al. (2018). The HAM10000 dataset. *Scientific Data*, 5(1).

---

> **Disclaimer:** This system is developed for **educational purposes only** and must **not** be used for actual medical diagnosis. Any clinical application would require extensive validation, regulatory approval, and integration with established healthcare workflows. Always consult a qualified dermatologist for medical decisions.

---

*Course: Medical Informatics | Report Generated: December 2024*
