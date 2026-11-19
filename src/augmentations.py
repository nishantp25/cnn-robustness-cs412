"""
Data augmentation for Model B (the augmented CNN).

build_augmentation() returns a block of Keras preprocessing layers that is
placed at the start of the CNN (see models.build_cnn_model). These layers:
    - apply a NEW random transformation to every image in every batch
    - are only active during training (model.fit)
    - do nothing during model.evaluate() / model.predict()

All transformations are mild so every image still clearly shows its class.

Run this file directly to save a grid of example augmented images:
    python -m src.augmentations
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ---- Augmentation strengths (kept mild on purpose) ----
ROTATION_FACTOR = 15 / 360   # up to +/- 15 degrees (Keras uses fractions of a full turn)
TRANSLATION_FACTOR = 0.1     # shift up to 10% of width/height (~3 pixels)
BRIGHTNESS_FACTOR = 0.2      # brighten/darken by up to 0.2 on the [0, 1] scale
CONTRAST_FACTOR = 0.2        # change contrast by up to +/- 20%
NOISE_STDDEV = 0.03          # light Gaussian noise (std 0.03 on the [0, 1] scale)


def build_augmentation(seed=42):
    """
    Return a keras.Sequential of random augmentation layers.

    Input/output: float32 images in [0, 1], shape (batch, 32, 32, 3).
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(ROTATION_FACTOR, fill_mode="reflect", seed=seed),
            layers.RandomTranslation(TRANSLATION_FACTOR, TRANSLATION_FACTOR,
                                     fill_mode="reflect", seed=seed),
            layers.RandomBrightness(BRIGHTNESS_FACTOR, value_range=(0.0, 1.0), seed=seed),
            layers.RandomContrast(CONTRAST_FACTOR, value_range=(0.0, 1.0), seed=seed),
            layers.GaussianNoise(NOISE_STDDEV, seed=seed),
        ],
        name="augmentation",
    )


def save_augmentation_examples(x, y, path, n_images=6, n_versions=5, seed=42):
    """
    Save a grid: each row is one training image (original in the first
    column) followed by several random augmented versions of it.
    """
    from src.data import get_class_name

    augment = build_augmentation(seed=seed)
    fig, axes = plt.subplots(n_images, n_versions + 1,
                             figsize=(1.6 * (n_versions + 1), 1.7 * n_images))

    for row in range(n_images):
        img = x[row:row + 1]
        axes[row, 0].imshow(img[0])
        axes[row, 0].set_title(f"original\n{get_class_name(y[row])}", fontsize=8)
        for col in range(1, n_versions + 1):
            aug = augment(img, training=True).numpy()[0]
            axes[row, col].imshow(np.clip(aug, 0.0, 1.0))
            axes[row, col].set_title(f"augmented {col}", fontsize=8)

    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Augmentation examples saved to: {path}")


def _sanity_check():
    from src.data import load_cifar10

    (x_train, y_train), _ = load_cifar10()
    sample = x_train[:8]

    augment = build_augmentation()
    out_train = augment(sample, training=True).numpy()
    out_test = augment(sample, training=False).numpy()

    print(f"Input shape:  {sample.shape}")
    print(f"Output shape: {out_train.shape}")
    print(f"Training mode changes images:   {not np.allclose(out_train, sample)}")
    print(f"Inference mode leaves them alone: {np.allclose(out_test, sample)}")
    print(f"Augmented pixel range: {out_train.min():.2f} to {out_train.max():.2f}")

    assert out_train.shape == sample.shape
    assert not np.allclose(out_train, sample)
    assert np.allclose(out_test, sample)

    save_augmentation_examples(
        x_train, y_train,
        os.path.join("results", "figures", "augmentation_examples.png"))
    print("\nAll checks passed.")


if __name__ == "__main__":
    _sanity_check()