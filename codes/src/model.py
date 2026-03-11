"""
Skin Cancer Detection System - Model Architecture
Binary classification model based on MobileNetV2 with Transfer Learning
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, 
    GlobalAveragePooling2D, 
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from src.config import INPUT_SHAPE, LEARNING_RATE, MODEL_PATH


def build_model(input_shape: tuple = INPUT_SHAPE, trainable_base: bool = False) -> Model:
    """
    Builds a MobileNetV2-based Transfer Learning model.
    
    Args:
        input_shape: Input image size (224, 224, 3)
        trainable_base: Whether base model layers are trainable
    
    Returns:
        Model: Compiled Keras model
    """
    # Pre-trained MobileNetV2 base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Base model katmanlarını dondur/aç
    base_model.trainable = trainable_base
    
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
    
    # Model oluştur
    model = Model(inputs=base_model.input, outputs=output, name='SkinCancerDetector')
    
    return model


def compile_model(model: Model, learning_rate: float = LEARNING_RATE) -> Model:
    """
    Compiles the model.
    
    Args:
        model: Keras model
        learning_rate: Learning rate
    
    Returns:
        Model: Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def unfreeze_model(model: Model, num_layers_to_unfreeze: int = 30) -> Model:
    """
    Unfreezes the last N layers of the base model for fine-tuning.
    
    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze
    
    Returns:
        Model: Updated model
    """
    # Find base model (first layer is MobileNetV2)
    base_model = model.layers[0] if hasattr(model.layers[0], 'layers') else None
    
    if base_model is None:
        # Alternative method
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower():
                base_model = layer
                break
    
    if base_model is None:
        print("Base model not found, unfreezing all layers...")
        model.trainable = True
        return model
    
    # Make base model trainable
    base_model.trainable = True
    
    # Freeze all except last N layers
    total_layers = len(base_model.layers)
    freeze_until = total_layers - num_layers_to_unfreeze
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
    
    print(f"Base model: {total_layers} layers")
    print(f"Frozen: {freeze_until} layers")
    print(f"Trainable: {num_layers_to_unfreeze} layers")
    
    return model


def get_model_summary(model: Model) -> str:
    """
    Returns model summary as string.
    
    Args:
        model: Keras model
    
    Returns:
        str: Model summary
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


def save_model(model: Model, path: str = MODEL_PATH):
    """
    Saves the model.
    
    Args:
        model: Model to save
        path: Save path
    """
    model.save(path)
    print(f"Model saved: {path}")


def load_trained_model(path: str = MODEL_PATH) -> Model:
    """
    Loads the trained model.
    
    Args:
        path: Path to the model file
    
    Returns:
        Model: Loaded model
    """
    model = load_model(path)
    print(f"Model loaded: {path}")
    return model


if __name__ == "__main__":
    # Test
    print("Building model...")
    model = build_model()
    model = compile_model(model)
    
    print("\nModel Summary:")
    model.summary()
    
    # Trainable/Non-trainable parameter counts
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"\nTotal Parameters: {trainable_params + non_trainable_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Frozen: {non_trainable_params:,}")
