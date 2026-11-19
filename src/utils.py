"""
Shared training utilities used by BOTH train_baseline.py and train_augmented.py.

Keeping the training loop in one place guarantees that the baseline and
augmented models use identical settings (optimizer, batch size, epoch limit,
early stopping, validation set). The only thing that differs between the two
runs is whether the model contains an augmentation block.
"""

import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ---- Training configuration (shared by both models) ----
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
VAL_FRACTION = 0.1

# ---- Output folders ----
MODELS_DIR = "saved_models"
HISTORY_DIR = os.path.join("results", "histories")
FIGURES_DIR = os.path.join("results", "figures")
TABLES_DIR = os.path.join("results", "tables")


def set_seed(seed=SEED):
    """Fix random seeds (Python, NumPy, TensorFlow) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def ensure_output_dirs():
    for d in (MODELS_DIR, HISTORY_DIR, FIGURES_DIR, TABLES_DIR):
        os.makedirs(d, exist_ok=True)


def train_model(model, run_name, x_tr, y_tr, x_val, y_val,
                epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
    """
    Train a compiled model with EarlyStopping + ModelCheckpoint.

    The best model (lowest validation loss) is saved to
    saved_models/<run_name>.keras, and the training history to
    results/histories/<run_name>_history.csv.

    Returns:
        history_df: pandas DataFrame with loss/accuracy per epoch.
    """
    ensure_output_dirs()
    model_path = os.path.join(MODELS_DIR, f"{run_name}.keras")

    callbacks = [
        # Stop when validation loss has not improved for PATIENCE epochs,
        # and roll the weights back to the best epoch.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save the model every time validation loss reaches a new best.
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,  # one line per epoch
    )

    history_df = pd.DataFrame(history.history)
    history_df.index = history_df.index + 1
    history_df.index.name = "epoch"
    history_path = os.path.join(HISTORY_DIR, f"{run_name}_history.csv")
    history_df.to_csv(history_path)

    print(f"\nBest model saved to:  {model_path}")
    print(f"Training history saved to: {history_path}")
    return history_df


def plot_training_curves(history_df, run_name, title):
    """Save training-vs-validation accuracy and loss plots side by side."""
    ensure_output_dirs()
    epochs = history_df.index

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_acc.plot(epochs, history_df["accuracy"], label="Training", marker="o", markersize=3)
    ax_acc.plot(epochs, history_df["val_accuracy"], label="Validation", marker="o", markersize=3)
    ax_acc.set_title(f"{title}: Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    ax_loss.plot(epochs, history_df["loss"], label="Training", marker="o", markersize=3)
    ax_loss.plot(epochs, history_df["val_loss"], label="Validation", marker="o", markersize=3)
    ax_loss.set_title(f"{title}: Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Sparse categorical cross-entropy")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"{run_name}_training_curves.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to:  {fig_path}")
    return fig_path


def evaluate_clean(model, x_test, y_test, run_name):
    """Evaluate on the clean test set and save the result as JSON."""
    ensure_output_dirs()
    loss, acc = model.evaluate(x_test, y_test, batch_size=256, verbose=0)
    result = {"model": run_name, "clean_test_loss": float(loss),
              "clean_test_accuracy": float(acc)}
    path = os.path.join(TABLES_DIR, f"{run_name}_clean_test.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Clean test accuracy: {acc:.4f}  (saved to {path})")
    return result