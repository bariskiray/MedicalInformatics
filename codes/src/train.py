"""
Skin Cancer Detection System - Training Script
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    EPOCHS,
    BATCH_SIZE,
    MODEL_PATH,
    MODEL_DIR,
    METRICS_PATH,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    CLASS_NAMES,
    PREDICTION_THRESHOLD
)
from src.data_loader import prepare_dataset, calculate_class_weights, get_class_distribution
from src.preprocessing import load_all_images, split_data, create_data_generators, create_balanced_generator
from src.model import build_model, compile_model, unfreeze_model, save_model


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


def plot_training_history(history, save_path: str = None):
    """
    Visualizes training history.
    
    Args:
        history: Keras training history
        save_path: Save path (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training graph saved: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """
    Visualizes confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Save path (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path: str = None):
    """
    Plots ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        save_path: Save path (optional)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved: {save_path}")
    
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions (using threshold)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > PREDICTION_THRESHOLD).astype(int).flatten()
    
    # Test loss and metrics
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    print(f"Test AUC: {results[4]:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualizations
    plot_confusion_matrix(y_test, y_pred, os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_pred_proba, os.path.join(MODEL_DIR, 'roc_curve.png'))
    
    # Get detailed metrics from classification report
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
    
    # Save metrics to JSON
    metrics = {
        'test_loss': float(results[0]),
        'test_accuracy': float(results[1]),
        'test_precision': float(results[2]),
        'test_recall': float(results[3]),
        'test_auc': float(results[4]),
        'benign': {
            'precision': float(precision[0]),
            'recall': float(recall[0]),
            'f1_score': float(f1[0])
        },
        'melanoma': {
            'precision': float(precision[1]),
            'recall': float(recall[1]),
            'f1_score': float(f1[1])
        },
        'confusion_matrix': {
            'true_negative': int(cm[0][0]),
            'false_positive': int(cm[0][1]),
            'false_negative': int(cm[1][0]),
            'true_positive': int(cm[1][1])
        },
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to JSON file
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nMetrics saved: {METRICS_PATH}")
    
    return results


def main():
    """
    Main training function.
    """
    print("="*60)
    print("SKIN CANCER DETECTION SYSTEM - MODEL TRAINING")
    print("="*60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU found: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("\nGPU not found, training will be performed on CPU.")
    
    # 1. Data Loading
    print("\n" + "-"*40)
    print("STEP 1: Data Loading")
    print("-"*40)
    
    image_paths, labels, df = prepare_dataset()
    print(f"Total images: {len(image_paths)}")
    
    distribution = get_class_distribution(labels)
    print(f"\nClass Distribution:")
    print(f"  Benign: {distribution['benign_count']} ({distribution['benign_ratio']:.2%})")
    print(f"  Melanoma: {distribution['melanoma_count']} ({distribution['melanoma_ratio']:.2%})")
    
    # Calculate class weights
    class_weights = calculate_class_weights(labels)
    print(f"\nClass Weights: {class_weights}")
    
    # 2. Load and Process Images
    print("\n" + "-"*40)
    print("STEP 2: Image Processing")
    print("-"*40)
    
    print("Loading images (this may take a few minutes)...")
    X, y = load_all_images(image_paths, labels)
    print(f"Image array shape: {X.shape}")
    print(f"Label array shape: {y.shape}")
    
    # 3. Data Splitting
    print("\n" + "-"*40)
    print("STEP 3: Data Splitting")
    print("-"*40)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 4. Data Augmentation - Balanced Generator
    print("\n" + "-"*40)
    print("STEP 4: Data Augmentation (Balanced)")
    print("-"*40)
    
    # Normal generator for validation
    _, val_datagen = create_data_generators()
    
    # Balanced generator for training (aggressive augmentation for melanoma)
    train_generator, steps_per_epoch = create_balanced_generator(
        X_train, y_train, 
        target_ratio=1.0  # Equalize melanoma count to benign
    )
    print(f"Balanced generator created.")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # 5. Model Creation
    print("\n" + "-"*40)
    print("STEP 5: Model Creation")
    print("-"*40)
    
    model = build_model()
    model = compile_model(model)
    model.summary()
    
    # 6. Phase 1: Feature Extraction Training
    print("\n" + "-"*40)
    print("STEP 6: Phase 1 - Feature Extraction Training")
    print("-"*40)
    
    callbacks = create_callbacks(MODEL_PATH)
    
    # Training with balanced generator (aggressive augmentation for melanoma)
    # NOTE: class_weight is not supported with Python generators, balanced generator is already balanced
    history1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nPhase 1 completed.")
    
    # 7. Phase 2: Fine-tuning
    print("\n" + "-"*40)
    print("STEP 7: Phase 2 - Fine-tuning")
    print("-"*40)
    
    # Unfreeze last 30 layers
    model = unfreeze_model(model, num_layers_to_unfreeze=30)
    
    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
    
    # Fine-tune callbacks
    fine_tune_callbacks = create_callbacks(MODEL_PATH)
    
    # Use balanced generator for fine-tuning as well
    # NOTE: class_weight is not supported with Python generators, balanced generator is already balanced
    history2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=fine_tune_callbacks,
        verbose=1
    )
    
    print("\nPhase 2 (Fine-tuning) completed.")
    
    # 8. Model Evaluation
    print("\n" + "-"*40)
    print("STEP 8: Model Evaluation")
    print("-"*40)
    
    evaluate_model(model, X_test, y_test)
    
    # 9. Save Training Graphs
    # Combine history from both phases
    combined_history = {
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'precision': history1.history['precision'] + history2.history['precision'],
        'val_precision': history1.history['val_precision'] + history2.history['val_precision'],
        'recall': history1.history['recall'] + history2.history['recall'],
        'val_recall': history1.history['val_recall'] + history2.history['val_recall']
    }
    
    class CombinedHistory:
        def __init__(self, h):
            self.history = h
    
    plot_training_history(
        CombinedHistory(combined_history), 
        os.path.join(MODEL_DIR, 'training_history.png')
    )
    
    # 10. Save final model
    save_model(model, MODEL_PATH)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Model saved: {MODEL_PATH}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
