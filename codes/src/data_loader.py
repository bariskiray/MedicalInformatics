"""
Skin Cancer Detection System - Data Loading Module
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, List

from src.config import (
    METADATA_PATH,
    IMAGES_PART1_DIR,
    IMAGES_PART2_DIR,
    MALIGNANT_LABELS,
    BENIGN_LABELS,
    MELANOMA_WEIGHT_MULTIPLIER
)


def load_metadata() -> pd.DataFrame:
    """
    Loads the HAM10000 metadata CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing metadata
    """
    df = pd.read_csv(METADATA_PATH)
    return df


def get_image_path(image_id: str) -> str:
    """
    Finds the full path of the image file for the given image_id.
    
    Args:
        image_id: Image identifier (e.g., 'ISIC_0027419')
    
    Returns:
        str: Full path to the image file
    """
    filename = f"{image_id}.jpg"
    
    # Check part_1 first
    path1 = os.path.join(IMAGES_PART1_DIR, filename)
    if os.path.exists(path1):
        return path1
    
    # Then check part_2
    path2 = os.path.join(IMAGES_PART2_DIR, filename)
    if os.path.exists(path2):
        return path2
    
    raise FileNotFoundError(f"Image not found: {image_id}")


def convert_to_binary_label(dx: str) -> int:
    """
    Converts diagnosis (dx) to binary label.
    
    Args:
        dx: Diagnosis code (e.g., 'mel', 'nv', 'bkl')
    
    Returns:
        int: 1 = Melanoma (Malignant), 0 = Benign
    """
    if dx in MALIGNANT_LABELS:
        return 1  # Melanoma
    elif dx in BENIGN_LABELS:
        return 0  # Benign
    else:
        raise ValueError(f"Unknown diagnosis code: {dx}")


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


def get_class_distribution(labels: np.ndarray) -> dict:
    """
    Calculates class distribution.
    
    Args:
        labels: Binary labels array
    
    Returns:
        dict: Class counts and ratios
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    distribution = {
        'benign_count': int(counts[0]) if 0 in unique else 0,
        'melanoma_count': int(counts[1]) if 1 in unique else 0,
        'total': total,
        'benign_ratio': counts[0] / total if 0 in unique else 0,
        'melanoma_ratio': counts[1] / total if 1 in unique else 0
    }
    
    return distribution


def calculate_class_weights(labels: np.ndarray, melanoma_multiplier: float = MELANOMA_WEIGHT_MULTIPLIER) -> dict:
    """
    Computes class weights for imbalanced dataset.
    Adds extra weight to melanoma class.
    
    Args:
        labels: Binary labels array
        melanoma_multiplier: Weight multiplier for melanoma class (default from config)
    
    Returns:
        dict: Class weights {0: weight_benign, 1: weight_melanoma}
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Inversely proportional weights
    weights = {}
    for cls, count in zip(unique, counts):
        base_weight = total / (len(unique) * count)
        
        # Give extra weight to melanoma class (1)
        if int(cls) == 1:  # Melanoma
            weights[int(cls)] = base_weight * melanoma_multiplier
        else:  # Benign
            weights[int(cls)] = base_weight
    
    return weights


if __name__ == "__main__":
    # Test
    print("Preparing dataset...")
    image_paths, labels, df = prepare_dataset()
    
    print(f"\nTotal number of images: {len(image_paths)}")
    
    distribution = get_class_distribution(labels)
    print(f"\nClass Distribution:")
    print(f"  Benign: {distribution['benign_count']} ({distribution['benign_ratio']:.2%})")
    print(f"  Melanoma: {distribution['melanoma_count']} ({distribution['melanoma_ratio']:.2%})")
    
    weights = calculate_class_weights(labels)
    print(f"\nClass Weights:")
    print(f"  Benign (0): {weights[0]:.4f}")
    print(f"  Melanoma (1): {weights[1]:.4f}")
