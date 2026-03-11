"""
Skin Cancer Detection System - Configuration File
"""
import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Skin Cancer MNIST Archive")
IMAGES_PART1_DIR = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMAGES_PART2_DIR = os.path.join(DATA_DIR, "HAM10000_images_part_2")
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_model.h5")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# Image Parameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_EPOCHS = 10

# Data Split Ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Class Labels
CLASS_NAMES = ['Benign', 'Melanoma']

# dx values classified as Melanoma
MALIGNANT_LABELS = ['mel']  # Melanoma

# dx values classified as Benign
BENIGN_LABELS = ['bkl', 'nv', 'bcc', 'akiec', 'vasc', 'df']

# Random Seed (for reproducibility)
RANDOM_SEED = 42

# Class Weighting - Multiplier to give more weight to melanoma class
# NOTE: When using balanced generator, 1.0 is recommended (batches already balanced)
# Higher value = more emphasis on Melanoma
# With balanced generator: 1.0-1.3 recommended
MELANOMA_WEIGHT_MULTIPLIER = 1.0

# Data Augmentation - Aggressive augmentation parameters for Melanoma
# NOTE: Color changes WILL NOT BE USED (critical for medical diagnosis)
MELANOMA_ROTATION_RANGE = 45  # Rotation range for Melanoma (degrees)
MELANOMA_ZOOM_RANGE = 0.3  # Zoom range for Melanoma
MELANOMA_SHIFT_RANGE = 0.3  # Shift range for Melanoma (width/height)
MELANOMA_SHEAR_RANGE = 15  # Shear range for Melanoma (degrees)

# Normal augmentation parameters for Benign
BENIGN_ROTATION_RANGE = 20
BENIGN_ZOOM_RANGE = 0.2
BENIGN_SHIFT_RANGE = 0.2
BENIGN_SHEAR_RANGE = 10

# Prediction Threshold - Threshold for melanoma prediction
# Lower value = More melanoma detections (high recall, lower precision)
# Higher value = Fewer false positives (higher precision, lower recall)
# Suggested: 0.1-0.2 (very high recall), 0.2-0.3 (high recall), 0.5 (balanced)
PREDICTION_THRESHOLD = 0.3
