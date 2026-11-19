"""
CNN architecture for the CNN robustness project.

The SAME function builds both models:
    baseline  = build_cnn_model()
    augmented = build_cnn_model(augmentation=<augmentation layers>)

The only difference is the optional augmentation block at the very start.
Keras augmentation layers are only active during training; during
evaluate()/predict() they pass images through unchanged. So at test time
both models run exactly the same CNN layers on exactly the same images.

Run this file directly to print the architecture and run a quick check:
    python -m src.models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.data import IMAGE_SHAPE, NUM_CLASSES

LEARNING_RATE = 1e-3


def build_cnn_model(augmentation=None, learning_rate=LEARNING_RATE, name="cnn"):
    """
    Build and compile the CNN.

    Args:
        augmentation: optional Keras layer (or Sequential of layers) applied to
            the input during training only. None = baseline model.
        learning_rate: Adam learning rate.
        name: model name shown in model.summary().

    Returns:
        A compiled keras.Model.
        Input:  (batch, 32, 32, 3) float32 images in [0, 1]
        Output: (batch, 10) class probabilities that sum to 1
    """
    inputs = keras.Input(shape=IMAGE_SHAPE)
    x = inputs

    if augmentation is not None:
        x = augmentation(x)

    # Block 1: 32 filters learn simple features (edges, colors)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)          # 32x32 -> 16x16

    # Block 2: 64 filters combine edges into textures/parts
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)          # 16x16 -> 8x8

    # Block 3: 128 filters learn higher-level object parts
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)          # 8x8 -> 4x4

    # Classifier
    x = layers.Flatten()(x)                     # 4*4*128 = 2048 values
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)                  # reduces overfitting
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _sanity_check():
    """Build the model, print its summary, and check input/output shapes."""
    model = build_cnn_model(name="baseline_cnn")
    model.summary()

    dummy = np.random.rand(4, *IMAGE_SHAPE).astype("float32")
    probs = model.predict(dummy, verbose=0)

    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {probs.shape}")
    print(f"Each row sums to 1: {np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)}")

    assert probs.shape == (4, NUM_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    print("\nAll checks passed.")


if __name__ == "__main__":
    _sanity_check()