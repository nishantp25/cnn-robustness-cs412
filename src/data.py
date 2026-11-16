"""
CIFAR-10 data loading utilities for the CNN robustness project.

DATA CONTRACT (shared by training and evaluation code):
    - Images: float32 NumPy arrays, shape (N, 32, 32, 3), values in [0.0, 1.0]
    - Labels: int NumPy arrays, shape (N,), values 0-9 (see CLASS_NAMES)

Corruption functions should accept and return images in this same format
(float32, range [0, 1]) and must never change the labels.

Run this file directly to perform a quick sanity check:
    python -m src.data
"""

import numpy as np
import tensorflow as tf

# Index i in this list is the name of class label i.
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SHAPE = (32, 32, 3)


def normalize_images(images):
    """Convert uint8 pixels in [0, 255] to float32 values in [0, 1]."""
    return images.astype("float32") / 255.0


def load_cifar10(normalize=True):
    """
    Load the standard CIFAR-10 train and test sets.

    Args:
        normalize: if True, scale pixel values to [0, 1] as float32.

    Returns:
        (x_train, y_train), (x_test, y_test)
        x_train: (50000, 32, 32, 3)   y_train: (50000,)
        x_test:  (10000, 32, 32, 3)   y_test:  (10000,)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Keras gives labels with shape (N, 1); flatten them to (N,).
    y_train = y_train.flatten().astype("int64")
    y_test = y_test.flatten().astype("int64")

    if normalize:
        x_train = normalize_images(x_train)
        x_test = normalize_images(x_test)

    return (x_train, y_train), (x_test, y_test)


def split_train_validation(x_train, y_train, val_fraction=0.1, seed=42):
    """
    Split the training set into a smaller training set and a validation set.

    The split is random but fixed by `seed`, so the baseline and augmented
    models are validated on exactly the same images. The test set is never
    touched here.

    Returns:
        (x_tr, y_tr), (x_val, y_val)
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    n_val = int(len(x_train) * val_fraction)

    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]

    return (x_train[tr_idx], y_train[tr_idx]), (x_train[val_idx], y_train[val_idx])


def get_class_name(label):
    """Return the class name for an integer label (0-9)."""
    return CLASS_NAMES[int(label)]


def _sanity_check():
    """Print shapes, ranges, and class counts to confirm the data is correct."""
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    (x_tr, y_tr), (x_val, y_val) = split_train_validation(x_train, y_train)

    print("=== CIFAR-10 sanity check ===")
    print(f"x_train: {x_train.shape}, dtype={x_train.dtype}")
    print(f"y_train: {y_train.shape}, dtype={y_train.dtype}")
    print(f"x_test:  {x_test.shape}, dtype={x_test.dtype}")
    print(f"y_test:  {y_test.shape}, dtype={y_test.dtype}")
    print(f"Pixel range: min={x_train.min():.1f}, max={x_train.max():.1f}")
    print(f"Label range: {y_train.min()} to {y_train.max()}")
    print(f"After split -> train: {x_tr.shape[0]}, validation: {x_val.shape[0]}")

    print("\nTest images per class:")
    counts = np.bincount(y_test, minlength=NUM_CLASSES)
    for i, count in enumerate(counts):
        print(f"  {i}: {get_class_name(i):<10} {count}")

    # Automatic checks: fail loudly if anything is wrong.
    assert x_train.shape == (50000, *IMAGE_SHAPE)
    assert x_test.shape == (10000, *IMAGE_SHAPE)
    assert y_train.shape == (50000,) and y_test.shape == (10000,)
    assert 0.0 <= x_train.min() and x_train.max() <= 1.0
    assert set(np.unique(y_test)) == set(range(NUM_CLASSES))
    assert len(x_tr) + len(x_val) == len(x_train)
    print("\nAll checks passed.")


if __name__ == "__main__":
    _sanity_check()