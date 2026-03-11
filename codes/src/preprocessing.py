"""
Skin Cancer Detection System - Image Preprocessing Module
"""
import numpy as np
import cv2
from PIL import Image
from typing import Tuple
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    IMAGE_SIZE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    BATCH_SIZE,
    MELANOMA_ROTATION_RANGE,
    MELANOMA_ZOOM_RANGE,
    MELANOMA_SHIFT_RANGE,
    MELANOMA_SHEAR_RANGE,
    BENIGN_ROTATION_RANGE,
    BENIGN_ZOOM_RANGE,
    BENIGN_SHIFT_RANGE,
    BENIGN_SHEAR_RANGE
)


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


def load_image_for_prediction(image_path: str) -> np.ndarray:
    """
    Prepares a single image for prediction (adds batch dimension).
    
    Args:
        image_path: Path to the image file
    
    Returns:
        np.ndarray: Image ready for prediction (1, 224, 224, 3)
    """
    img = load_and_preprocess_image(image_path)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Processes an image uploaded from Streamlit.
    
    Args:
        uploaded_file: File from Streamlit file_uploader
    
    Returns:
        np.ndarray: Image ready for prediction (1, 224, 224, 3)
    """
    # Open with PIL
    img = Image.open(uploaded_file)
    
    # Convert to RGB (may be RGBA or L mode)
    img = img.convert('RGB')
    
    # Resize
    img = img.resize(IMAGE_SIZE)
    
    # Convert to NumPy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def load_all_images(image_paths: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads all images and applies preprocessing.
    
    Args:
        image_paths: Image file paths
        labels: Labels
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (images, labels) - processed images and labels
    """
    images = []
    valid_labels = []
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        try:
            img = load_and_preprocess_image(path)
            images.append(img)
            valid_labels.append(label)
            
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(image_paths)} images processed...")
                
        except Exception as e:
            print(f"Error: {path} could not be loaded - {e}")
            continue
    
    return np.array(images), np.array(valid_labels)


def split_data(
    X: np.ndarray, 
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into train, validation and test sets (stratified).
    
    Args:
        X: Images
        y: Labels
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split into train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=y
    )
    
    # Split temp into validation and test
    val_ratio = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=RANDOM_SEED,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_generators() -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Creates data generators for training and validation.
    
    Returns:
        Tuple[ImageDataGenerator, ImageDataGenerator]: (train_datagen, val_datagen)
    """
    # With augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # No augmentation for Validation/Test
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen


def create_train_generator(X_train: np.ndarray, y_train: np.ndarray, datagen: ImageDataGenerator):
    """
    Creates a generator for training data.
    
    Args:
        X_train: Training images
        y_train: Training labels
        datagen: ImageDataGenerator
    
    Returns:
        Generator: Generator that produces batches
    """
    return datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


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


if __name__ == "__main__":
    # Test
    print("Testing preprocessing module...")
    
    from src.data_loader import prepare_dataset
    
    print("\nPreparing dataset...")
    image_paths, labels, df = prepare_dataset()
    
    # Test with a small sample
    sample_paths = image_paths[:10]
    sample_labels = labels[:10]
    
    print(f"\nLoading 10 sample images...")
    images, valid_labels = load_all_images(sample_paths, sample_labels)
    
    print(f"\nImage shape: {images.shape}")
    print(f"Labels shape: {valid_labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
