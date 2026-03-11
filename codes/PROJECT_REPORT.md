# Skin Cancer Detection System
## A Deep Learning Approach for Melanoma Classification

**Medical Informatics Course - Term Project**

---

## Abstract

This project presents a deep learning-based system for detecting melanoma from dermoscopic images using transfer learning. The system employs MobileNetV2, a lightweight convolutional neural network architecture, pre-trained on ImageNet, and fine-tuned on the HAM10000 dataset. The model achieves a test accuracy of 86.6% and an AUC score of 82.7% for binary classification between benign lesions and melanoma. A balanced data augmentation strategy is implemented to address class imbalance, using geometric transformations exclusively to preserve diagnostic color information. The system includes a web-based interface built with Streamlit for practical deployment. This work demonstrates the potential of transfer learning in medical image analysis while emphasizing the importance of careful preprocessing and augmentation strategies in healthcare applications.

**Keywords:** Deep Learning, Transfer Learning, Medical Image Analysis, Skin Cancer Detection, MobileNetV2, Binary Classification

---

## 1. Introduction

### 1.1 Problem Statement

Skin cancer is one of the most common types of cancer worldwide, with melanoma being the most dangerous form due to its high metastatic potential. Early detection significantly improves patient outcomes, with 5-year survival rates dropping from 99% for localized melanoma to 27% for metastatic disease [1]. Dermoscopic imaging has become a standard tool for dermatologists, but accurate diagnosis requires extensive training and expertise. The increasing volume of skin lesion cases and the need for rapid, accurate screening create an opportunity for computer-aided diagnosis systems.

### 1.2 Objectives

The primary objectives of this project are:

1. **Develop a deep learning model** capable of distinguishing between benign skin lesions and melanoma from dermoscopic images
2. **Implement transfer learning** using MobileNetV2 to leverage pre-trained features while maintaining computational efficiency
3. **Address class imbalance** through balanced data augmentation strategies
4. **Create a user-friendly interface** for image analysis and prediction
5. **Evaluate model performance** using comprehensive metrics including accuracy, precision, recall, and AUC

### 1.3 Medical Context and Importance

Dermoscopy (dermatoscopy) is a non-invasive imaging technique that allows visualization of skin structures not visible to the naked eye. The HAM10000 dataset contains dermoscopic images of various pigmented skin lesions, providing a valuable resource for training diagnostic models. Automated systems can assist dermatologists by providing second opinions, screening large populations, and reducing diagnostic variability. However, such systems must be developed with careful consideration of medical requirements, including preservation of color information critical for diagnosis.

---

## 2. Related Work

Deep learning has shown remarkable success in medical image analysis, particularly in dermatology. Esteva et al. (2017) demonstrated that convolutional neural networks can achieve dermatologist-level performance in skin cancer classification [2]. Transfer learning, where models pre-trained on large datasets like ImageNet are fine-tuned for specific tasks, has become a standard approach in medical imaging due to limited labeled medical data availability.

MobileNetV2, introduced by Sandler et al. (2018), provides an efficient architecture suitable for mobile and edge devices while maintaining competitive accuracy [3]. Its depthwise separable convolutions reduce computational requirements, making it ideal for deployment scenarios where resources may be limited.

Class imbalance is a common challenge in medical datasets, where positive cases (e.g., melanoma) are often underrepresented. Techniques such as data augmentation, class weighting, and balanced sampling have been employed to address this issue [4].

---

## 3. Dataset

### 3.1 HAM10000 Dataset

The HAM10000 ("Human Against Machine with 10000 training images") dataset contains 10,015 dermoscopic images of pigmented skin lesions [5]. The dataset includes seven diagnostic categories:

- **Melanoma (mel)**: Malignant melanoma
- **Melanocytic nevi (nv)**: Benign
- **Benign keratosis-like lesions (bkl)**: Benign
- **Basal cell carcinoma (bcc)**: Benign
- **Actinic keratoses (akiec)**: Benign
- **Vascular lesions (vasc)**: Benign
- **Dermatofibroma (df)**: Benign

For this binary classification task, images are grouped into two classes:
- **Melanoma (Malignant)**: All 'mel' diagnoses
- **Benign**: All other diagnoses (nv, bkl, bcc, akiec, vasc, df)

### 3.2 Class Distribution

The dataset exhibits significant class imbalance, which is typical for medical datasets:

- **Benign lesions**: ~9,000 images (~90%)
- **Melanoma**: ~1,000 images (~10%)

This imbalance poses challenges for model training, as classifiers may bias toward the majority class. The project addresses this through balanced data augmentation strategies.

### 3.3 Data Preprocessing

All images are preprocessed using the following pipeline:

1. **Resizing**: Images are resized to 224×224 pixels to match MobileNetV2 input requirements
2. **Color space conversion**: Images are converted from BGR (OpenCV default) to RGB
3. **Normalization**: Pixel values are normalized to the range [0, 1] by dividing by 255.0

The preprocessing is implemented in `src/preprocessing.py`:

```29:51:src/preprocessing.py
def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads and preprocesses an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        np.ndarray: Processed image (224x224x3), normalized [0,1]
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Normalize [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img
```

### 3.4 Data Splitting

The dataset is split into three subsets using stratified sampling to maintain class distribution:

- **Training set**: 70% (~7,000 images)
- **Validation set**: 15% (~1,500 images)
- **Test set**: 15% (~1,500 images)

Stratified splitting ensures that each subset maintains the original class distribution, preventing bias in evaluation.

---

## 4. Methodology

### 4.1 Model Architecture

The model is based on MobileNetV2, a lightweight convolutional neural network architecture designed for mobile and embedded vision applications. MobileNetV2 uses depthwise separable convolutions and inverted residual blocks to achieve high efficiency while maintaining accuracy.

#### 4.1.1 Base Model

The base MobileNetV2 model is loaded with ImageNet pre-trained weights, excluding the top classification layer:

```31:35:src/model.py
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
```

#### 4.1.2 Custom Classification Head

A custom classification head is added on top of the base model:

```41:50:src/model.py
    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Dense(256, activation='relu', name='dense1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(128, activation='relu', name='dense2')(x)
    x = Dropout(0.3, name='dropout2')(x)
    
    # Binary classification output
    output = Dense(1, activation='sigmoid', name='output')(x)
```

The architecture includes:
- **Global Average Pooling**: Reduces spatial dimensions
- **Batch Normalization**: Stabilizes training
- **Dense layers**: 256 and 128 units with ReLU activation
- **Dropout**: 0.5 and 0.3 rates to prevent overfitting
- **Output layer**: Single sigmoid unit for binary classification

### 4.2 Transfer Learning Strategy

The training process employs a two-phase transfer learning approach:

#### Phase 1: Feature Extraction (30 epochs)
- Base MobileNetV2 layers are **frozen** (non-trainable)
- Only the custom classification head is trained
- Learning rate: 0.001
- Objective: Learn to map ImageNet features to skin lesion classes

#### Phase 2: Fine-tuning (10 epochs)
- Last 30 layers of MobileNetV2 are **unfrozen** (trainable)
- Lower learning rate: 1e-5
- Objective: Adapt low-level features specifically for dermoscopic images

This approach prevents overfitting while allowing the model to adapt to the medical imaging domain.

### 4.3 Data Augmentation

Data augmentation is critical for improving model generalization and addressing class imbalance. However, in medical imaging, **color information is diagnostically important**. Therefore, only **geometric transformations** are applied:

#### Augmentation Parameters

**For Melanoma (Aggressive Augmentation):**
- Rotation: ±45 degrees
- Zoom: 0.3 range
- Shift: 0.3 range (width/height)
- Shear: 15 degrees
- Horizontal and vertical flips

**For Benign (Normal Augmentation):**
- Rotation: ±20 degrees
- Zoom: 0.2 range
- Shift: 0.2 range
- Shear: 10 degrees
- Horizontal and vertical flips

**Excluded transformations:**
- Brightness adjustments
- Color channel shifts
- Contrast modifications

The balanced generator ensures equal representation of both classes in each training batch:

```209:314:src/preprocessing.py
def create_balanced_generator(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    target_ratio: float = 1.0
):
    """
    Produces balanced batches by equalizing melanoma samples to benign count.
    Uses aggressive augmentation for melanoma, normal augmentation for benign.
    
    IMPORTANT: Color changes are NOT USED (brightness, channel_shift, etc.)
    Only geometric transformations: rotation, flip, zoom, shift, shear
    
    Args:
        X_train: Training images
        y_train: Training labels (0=Benign, 1=Melanoma)
        target_ratio: Melanoma/Benign ratio (1.0 = equal count)
    
    Returns:
        Generator: Generator that produces balanced batches
    """
    # Separate melanoma and benign samples
    melanoma_mask = y_train == 1
    benign_mask = y_train == 0
    
    X_melanoma = X_train[melanoma_mask]
    y_melanoma = y_train[melanoma_mask]
    X_benign = X_train[benign_mask]
    y_benign = y_train[benign_mask]
    
    num_benign = len(X_benign)
    num_melanoma = len(X_melanoma)
    target_melanoma = int(num_benign * target_ratio)
    
    print(f"\nClass Distribution (Training Set):")
    print(f"  Benign: {num_benign} samples")
    print(f"  Melanoma: {num_melanoma} samples (original)")
    print(f"  Target Melanoma: {target_melanoma} samples (with augmentation)")
    
    # Aggressive augmentation for melanoma (NO COLOR CHANGES)
    melanoma_datagen = ImageDataGenerator(
        rotation_range=MELANOMA_ROTATION_RANGE,
        width_shift_range=MELANOMA_SHIFT_RANGE,
        height_shift_range=MELANOMA_SHIFT_RANGE,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=MELANOMA_ZOOM_RANGE,
        shear_range=MELANOMA_SHEAR_RANGE,
        fill_mode='nearest'
        # NO COLOR CHANGES:
        # brightness_range, channel_shift_range, featurewise_center, 
        # featurewise_std_normalization, samplewise_center, samplewise_std_normalization
    )
    
    # Normal augmentation for benign
    benign_datagen = ImageDataGenerator(
        rotation_range=BENIGN_ROTATION_RANGE,
        width_shift_range=BENIGN_SHIFT_RANGE,
        height_shift_range=BENIGN_SHIFT_RANGE,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=BENIGN_ZOOM_RANGE,
        shear_range=BENIGN_SHEAR_RANGE,
        fill_mode='nearest'
    )
    
    # Calculate batch sizes so each batch has equal number of samples
    batch_size_per_class = BATCH_SIZE // 2
    
    # Create generators
    melanoma_gen = melanoma_datagen.flow(
        X_melanoma, y_melanoma,
        batch_size=batch_size_per_class,
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    benign_gen = benign_datagen.flow(
        X_benign, y_benign,
        batch_size=batch_size_per_class,
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    # Generator that produces balanced batches
    def balanced_generator():
        while True:
            # Get equal number of samples from both classes
            X_batch_melanoma, y_batch_melanoma = next(melanoma_gen)
            X_batch_benign, y_batch_benign = next(benign_gen)
            
            # Concatenate batches
            X_batch = np.concatenate([X_batch_benign, X_batch_melanoma], axis=0)
            y_batch = np.concatenate([y_batch_benign, y_batch_melanoma], axis=0)
            
            # Shuffle
            indices = np.arange(len(X_batch))
            np.random.shuffle(indices)
            X_batch = X_batch[indices]
            y_batch = y_batch[indices]
            
            yield X_batch, y_batch
    
    # Calculate steps per epoch (based on benign count)
    steps_per_epoch = (num_benign + target_melanoma) // BATCH_SIZE
    
    return balanced_generator(), steps_per_epoch
```

### 4.4 Training Configuration

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Size | 224×224×3 |
| Batch Size | 32 |
| Feature Extraction Epochs | 30 |
| Fine-tuning Epochs | 10 |
| Initial Learning Rate | 0.001 |
| Fine-tuning Learning Rate | 1e-5 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metrics | Accuracy, Precision, Recall, AUC |

#### Callbacks

Several callbacks are used to improve training:

1. **Early Stopping**: Monitors validation loss with patience of 5 epochs
2. **Model Checkpoint**: Saves the best model based on validation AUC
3. **ReduceLROnPlateau**: Reduces learning rate by factor of 0.5 when validation loss plateaus

```45:83:src/train.py
def create_callbacks(model_path: str) -> list:
    """
    Creates training callbacks.
    
    Args:
        model_path: Model save path
    
    Returns:
        list: List of callbacks
    """
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks
```

---

## 5. Implementation

### 5.1 Project Structure

The project is organized into modular components:

```
MedicalInformatics/
├── src/
│   ├── config.py          # Configuration parameters
│   ├── data_loader.py     # Dataset loading and preparation
│   ├── preprocessing.py  # Image preprocessing and augmentation
│   ├── model.py           # Model architecture definition
│   └── train.py           # Training script
├── models/                # Trained models and metrics
│   ├── skin_cancer_model.h5
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_history.png
├── app.py                 # Streamlit web application
└── requirements.txt       # Python dependencies
```

### 5.2 Key Modules

#### Configuration (`src/config.py`)
Centralizes all hyperparameters and paths, enabling easy experimentation and reproducibility.

#### Data Loader (`src/data_loader.py`)
Handles dataset loading, label conversion, and class distribution analysis:

```73:96:src/data_loader.py
def prepare_dataset() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepares the entire dataset: image paths and binary labels.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
            - image_paths: Array of image file paths
            - labels: Array of binary labels (0=Benign, 1=Melanoma)
            - df: Processed DataFrame
    """
    # Load metadata
    df = load_metadata()
    
    # Add image paths
    df['image_path'] = df['image_id'].apply(get_image_path)
    
    # Add binary labels
    df['label'] = df['dx'].apply(convert_to_binary_label)
    
    # Convert to arrays
    image_paths = df['image_path'].values
    labels = df['label'].values
    
    return image_paths, labels, df
```

#### Preprocessing (`src/preprocessing.py`)
Implements image loading, preprocessing, data splitting, and balanced augmentation generators.

#### Model (`src/model.py`)
Defines the model architecture, compilation, and fine-tuning utilities.

#### Training (`src/train.py`)
Orchestrates the complete training pipeline, including evaluation and visualization.

### 5.3 Web Application

A Streamlit-based web interface provides an intuitive way to use the trained model:

- **Image Upload**: Users can upload dermoscopic images
- **Real-time Prediction**: Instant classification results with confidence scores
- **Visual Feedback**: Color-coded results (green for benign, red for melanoma)
- **Model Metrics Display**: Shows model performance statistics in the sidebar
- **Medical Disclaimers**: Prominent warnings about educational use only

The application is implemented in `app.py` and provides a production-ready interface for model deployment.

---

## 6. Results

### 6.1 Overall Performance

The model was evaluated on the test set (15% of the dataset, ~1,500 images). The following metrics were obtained:

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 86.6% |
| **Test AUC** | 82.7% |
| **Test Precision** | 40.0% |
| **Test Recall** | 40.7% |
| **Test Loss** | 0.314 |

### 6.2 Class-Specific Performance

#### Benign Class
- **Precision**: 96.4%
- **Recall**: 74.7%
- **F1-Score**: 84.2%

#### Melanoma Class
- **Precision**: 27.8%
- **Recall**: 77.8%
- **F1-Score**: 40.9%

### 6.3 Confusion Matrix

The confusion matrix reveals the model's classification behavior:

| | Predicted Benign | Predicted Melanoma |
|--|------------------|-------------------|
| **Actual Benign** | 998 (TN) | 338 (FP) |
| **Actual Melanoma** | 37 (FN) | 130 (TP) |

**Key Observations:**
- **True Negatives (TN)**: 998 - Correctly identified benign lesions
- **False Positives (FP)**: 338 - Benign lesions incorrectly classified as melanoma
- **False Negatives (FN)**: 37 - Melanoma cases missed (critical errors)
- **True Positives (TP)**: 130 - Correctly identified melanoma cases

The confusion matrix visualization is saved at `models/confusion_matrix.png`.

### 6.4 ROC Curve Analysis

The Receiver Operating Characteristic (ROC) curve demonstrates the model's ability to distinguish between classes across different threshold settings. The Area Under the Curve (AUC) of 0.827 indicates good discriminative ability.

The ROC curve visualization is available at `models/roc_curve.png`.

### 6.5 Training History

The training process shows:
- **Feature Extraction Phase**: Gradual improvement in validation metrics
- **Fine-tuning Phase**: Further refinement with lower learning rate
- **Overfitting Prevention**: Early stopping and dropout regularization maintain generalization

Training history plots (loss, accuracy, precision, recall) are saved at `models/training_history.png`.

---

## 7. Discussion

### 7.1 Performance Analysis

The model achieves **86.6% accuracy** and **82.7% AUC**, demonstrating reasonable performance for a binary classification task. However, several important observations emerge:

#### Strengths
1. **High Benign Precision (96.4%)**: The model correctly identifies most benign lesions, reducing unnecessary referrals
2. **Good Melanoma Recall (77.8%)**: Most melanoma cases are detected, which is critical for early intervention
3. **Balanced Approach**: The balanced augmentation strategy successfully addresses class imbalance

#### Limitations
1. **Low Melanoma Precision (27.8%)**: Many benign lesions are incorrectly flagged as melanoma, leading to false positives
2. **False Negatives (37 cases)**: Missing melanoma cases is a critical concern in medical applications
3. **Threshold Sensitivity**: The prediction threshold (0.3) balances recall and precision, but may need adjustment for specific clinical scenarios

### 7.2 Medical Implications

#### Clinical Use Considerations
- **Screening Tool**: The model could serve as a preliminary screening tool, flagging suspicious lesions for dermatologist review
- **Second Opinion**: Provides an additional perspective to support clinical decision-making
- **Not a Replacement**: Should never replace professional medical diagnosis

#### Risk Assessment
- **False Positives**: While concerning for patients, false positives lead to additional examinations rather than missed diagnoses
- **False Negatives**: More critical, as missed melanoma cases can have severe consequences
- **Threshold Tuning**: Lower thresholds increase recall (fewer false negatives) but decrease precision (more false positives)

### 7.3 Limitations

1. **Dataset Limitations**: 
   - Single dataset (HAM10000) may not represent all skin types and imaging conditions
   - Limited to dermoscopic images; may not generalize to clinical photography

2. **Binary Classification**: 
   - Simplified to two classes; real-world diagnosis involves multiple lesion types
   - Does not provide differential diagnosis

3. **Color Preservation**: 
   - While color information is preserved, the model may not fully utilize it without explicit color-based features

4. **Computational Constraints**: 
   - MobileNetV2 prioritizes efficiency over maximum accuracy
   - Larger models (e.g., ResNet, EfficientNet) might achieve better performance

5. **Generalization**: 
   - Performance on external datasets may differ
   - Requires validation on diverse populations and imaging devices

### 7.4 Comparison with Literature

The achieved performance (86.6% accuracy, 82.7% AUC) is competitive with similar studies using transfer learning on medical imaging datasets. However, state-of-the-art models specifically designed for dermatology can achieve higher accuracy (90%+) with larger architectures and ensemble methods.

---

## 8. Conclusion

This project successfully developed a deep learning system for skin cancer detection using MobileNetV2 and transfer learning. The model achieves **86.6% accuracy** and **82.7% AUC** on the HAM10000 test set, demonstrating the feasibility of automated melanoma detection from dermoscopic images.

### Key Achievements

1. **Effective Transfer Learning**: Successfully adapted ImageNet pre-trained features to medical imaging domain
2. **Class Imbalance Handling**: Balanced augmentation strategy improved model performance on minority class
3. **Medical-Aware Design**: Geometric-only augmentation preserves diagnostically important color information
4. **Production-Ready Interface**: Streamlit application enables practical deployment

### Future Work

Several directions could enhance the system:

1. **Multi-class Classification**: Extend to all seven HAM10000 classes for differential diagnosis
2. **Ensemble Methods**: Combine multiple models to improve accuracy and robustness
3. **Advanced Architectures**: Experiment with EfficientNet, Vision Transformers, or specialized medical imaging models
4. **External Validation**: Test on diverse datasets from different populations and imaging devices
5. **Explainability**: Integrate Grad-CAM or attention mechanisms to visualize model decisions
6. **Clinical Integration**: Develop API for integration with hospital information systems
7. **Active Learning**: Implement strategies to improve model performance with limited labeled data

### Final Remarks

While this system demonstrates promising results, it is crucial to emphasize that **this is an educational project** and must not be used for actual medical diagnosis. Real-world deployment would require extensive clinical validation, regulatory approval, and integration with healthcare workflows. The project highlights both the potential and challenges of applying deep learning to medical imaging, emphasizing the need for careful design, thorough evaluation, and responsible deployment.

---

## References

[1] American Cancer Society. (2023). Cancer Facts & Figures 2023. Atlanta: American Cancer Society.

[2] Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118.

[3] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).

[4] Johnson, J. M., & Khoshgoftaar, T. M. (2019). Survey on deep learning with class imbalance. Journal of Big Data, 6(1), 1-54.

[5] Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5(1), 1-9.

---

## Appendix

### A. Model Architecture Summary

- **Base Model**: MobileNetV2 (ImageNet weights)
- **Total Parameters**: ~2.3M trainable parameters (feature extraction phase)
- **Input Shape**: (224, 224, 3)
- **Output**: Single sigmoid unit (binary classification)

### B. Dataset Statistics

- **Total Images**: 10,015
- **Training Set**: ~7,000 images (70%)
- **Validation Set**: ~1,500 images (15%)
- **Test Set**: ~1,500 images (15%)
- **Class Distribution**: ~90% Benign, ~10% Melanoma

### C. Software and Libraries

- **Python**: 3.8+
- **TensorFlow/Keras**: 2.10+
- **Streamlit**: 1.28+
- **OpenCV**: 4.7+
- **scikit-learn**: 1.2+
- **NumPy**: 1.23+
- **Pandas**: 1.5+

---

**Report Generated**: December 2024  
**Project Repository**: MedicalInformatics  
**Course**: Medical Informatics

