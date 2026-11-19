"""
Train Model A: the baseline CNN (no data augmentation).

Full run:
    python -m src.train_baseline

Quick test run (small subset, 1 epoch, saved under a separate name):
    python -m src.train_baseline --quick
"""

import argparse

from src.data import load_cifar10, split_train_validation
from src.models import build_cnn_model
from src.utils import (SEED, VAL_FRACTION, MAX_EPOCHS, set_seed, train_model,
                       plot_training_curves, evaluate_clean)


def main():
    parser = argparse.ArgumentParser(description="Train the baseline CNN.")
    parser.add_argument("--quick", action="store_true",
                        help="1 epoch on 5,000 images to test the pipeline.")
    args = parser.parse_args()

    set_seed(SEED)

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    (x_tr, y_tr), (x_val, y_val) = split_train_validation(
        x_train, y_train, val_fraction=VAL_FRACTION, seed=SEED)

    if args.quick:
        run_name, epochs = "baseline_quicktest", 1
        x_tr, y_tr = x_tr[:5000], y_tr[:5000]
    else:
        run_name, epochs = "baseline", MAX_EPOCHS

    print(f"Run: {run_name} | train={len(x_tr)}, val={len(x_val)}, "
          f"test={len(x_test)} | max epochs={epochs}")

    model = build_cnn_model(augmentation=None, name="baseline_cnn")

    history_df = train_model(model, run_name, x_tr, y_tr, x_val, y_val, epochs=epochs)
    plot_training_curves(history_df, run_name, title="Baseline CNN")

    # Final, one-time check on the untouched test set.
    evaluate_clean(model, x_test, y_test, run_name)


if __name__ == "__main__":
    main()