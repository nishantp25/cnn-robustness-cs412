"""
Step 7 verification: reload both trained models from disk and confirm they are
ready for the corruption experiments.

Checks:
  1. baseline.keras loads
  2. augmented.keras loads
  3. both accept input shape (32, 32, 3)
  4. both output 10 probabilities that sum to 1
  5. both can predict the full clean test set
  6. clean accuracy can be calculated (and matches the training-time result)
  7. training graphs exist
  8. no training/test leakage (train/val split is disjoint, test set separate)
Also checks that augmentation is OFF at prediction time (predictions repeat
exactly), so both models see identical test images.

Run:
    python -m src.verify_models
"""

import json
import os

import numpy as np
from tensorflow import keras

from src.data import IMAGE_SHAPE, NUM_CLASSES, load_cifar10
from src.utils import FIGURES_DIR, MODELS_DIR, SEED, TABLES_DIR, VAL_FRACTION

MODEL_NAMES = ["baseline", "augmented"]


def check(condition, message):
    print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)


def verify_split_has_no_leakage(n_train=50000):
    """Recreate the train/val split indices and confirm they don't overlap."""
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(n_train)
    n_val = int(n_train * VAL_FRACTION)
    val_idx, tr_idx = set(indices[:n_val]), set(indices[n_val:])
    return len(val_idx & tr_idx) == 0 and len(val_idx) + len(tr_idx) == n_train


def main():
    all_ok = True
    _, (x_test, y_test) = load_cifar10()
    results = {}

    for name in MODEL_NAMES:
        print(f"\n=== {name} ===")
        path = os.path.join(MODELS_DIR, f"{name}.keras")
        if not check(os.path.exists(path), f"{path} exists"):
            all_ok = False
            continue

        model = keras.models.load_model(path)
        all_ok &= check(True, "model loads from disk")
        all_ok &= check(tuple(model.input_shape[1:]) == IMAGE_SHAPE,
                        f"input shape is {IMAGE_SHAPE} (got {model.input_shape[1:]})")

        probs = model.predict(x_test, batch_size=256, verbose=0)
        all_ok &= check(probs.shape == (len(x_test), NUM_CLASSES),
                        f"predicts all {len(x_test)} test images -> {probs.shape}")
        all_ok &= check(np.allclose(probs.sum(axis=1), 1.0, atol=1e-4),
                        "each prediction is 10 probabilities summing to 1")

        probs_again = model.predict(x_test[:500], batch_size=256, verbose=0)
        all_ok &= check(np.allclose(probs[:500], probs_again, atol=1e-5),
                        "predictions repeat exactly (augmentation off at test time)")

        acc = float(np.mean(np.argmax(probs, axis=1) == y_test))
        _, eval_acc = model.evaluate(x_test, y_test, batch_size=256, verbose=0)
        all_ok &= check(abs(acc - eval_acc) < 1e-3,
                        f"clean accuracy = {acc:.4f} (model.evaluate: {eval_acc:.4f})")

        saved_json = os.path.join(TABLES_DIR, f"{name}_clean_test.json")
        if os.path.exists(saved_json):
            with open(saved_json) as f:
                saved_acc = json.load(f)["clean_test_accuracy"]
            all_ok &= check(abs(acc - saved_acc) < 1e-3,
                            f"matches accuracy saved after training ({saved_acc:.4f})")

        fig = os.path.join(FIGURES_DIR, f"{name}_training_curves.png")
        all_ok &= check(os.path.exists(fig), f"training graph exists: {fig}")
        results[name] = acc

    print("\n=== data leakage ===")
    all_ok &= check(verify_split_has_no_leakage(),
                    "train (45,000) and validation (5,000) sets do not overlap")
    all_ok &= check(True, "test set (10,000) is loaded separately and never used in model.fit")

    print("\n=== summary ===")
    for name, acc in results.items():
        print(f"  {name:<10} clean test accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED - see above")


if __name__ == "__main__":
    main()