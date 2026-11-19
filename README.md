# Improving CNN Robustness to Common Image Corruptions Using Data Augmentation

CS 412 – Introduction to Machine Learning, Final Project
Team: Nishant Patel, Mahi Patel

## Research Question
Does data augmentation improve the robustness of a Convolutional Neural Network
to common image corruptions?

## Experiment
- **Dataset:** CIFAR-10 (50,000 train / 10,000 test, 32×32 RGB, 10 classes)
- **Model A – Baseline CNN:** trained on standard training images
- **Model B – Augmented CNN:** same architecture, trained with data augmentation
  (horizontal flip, ±15° rotation, ±10% translation, brightness ±0.2,
  contrast ±20%, Gaussian noise σ=0.03)
- **Test conditions:** clean, Gaussian noise, blur, rotation, brightness
  (and contrast if time permits), at multiple severity levels
- Both models are evaluated on the exact same clean and corrupted test sets.

## Setup
Requires Python 3.10–3.13 (TensorFlow does not support newer versions yet).

```bash
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

All commands are run from the repository root.

## Training (Member 1)
```bash
python -m src.data               # sanity-check CIFAR-10 loading
python -m src.models             # print the CNN architecture
python -m src.train_baseline     # trains Model A -> saved_models/baseline.keras
python -m src.augmentations      # saves example augmented images
python -m src.train_augmented    # trains Model B -> saved_models/augmented.keras
python -m src.verify_models      # reloads both models and checks them
```
Add `--quick` to either training script for a 1-epoch smoke test.

Shared training settings (`src/utils.py`): Adam (lr 0.001), sparse categorical
cross-entropy, batch size 64, up to 50 epochs, EarlyStopping on validation loss
(patience 5, best weights restored), seed 42, fixed 45,000 / 5,000
train/validation split.

## Integration Guide for Corruption / Evaluation Code (Member 2)

### Data contract
| Item | Format |
|---|---|
| Images | `float32` NumPy array, shape `(N, 32, 32, 3)`, values in `[0, 1]` |
| Labels | `int64` NumPy array, shape `(N,)`, values `0–9` |
| Class names | `src.data.CLASS_NAMES` (index = label) |

Corruption functions should:
1. take and return images in the format above (clip outputs to `[0, 1]`)
2. never change the labels
3. be deterministic: use a fixed random seed (e.g. for Gaussian noise), so the
   same corrupted test set can be regenerated
4. be applied **once** per condition, with the **same** corrupted array passed to
   both models

### Functions available from Member 1's code
```python
from src.data import load_cifar10, CLASS_NAMES, NUM_CLASSES, get_class_name
(x_train, y_train), (x_test, y_test) = load_cifar10()   # normalized to [0, 1]
```

### Example evaluation
```python
from tensorflow import keras
from src.data import load_cifar10
from src.corruptions import add_gaussian_noise   # Member 2's function

_, (x_test, y_test) = load_cifar10()
baseline = keras.models.load_model("saved_models/baseline.keras")
augmented = keras.models.load_model("saved_models/augmented.keras")

x_noisy = add_gaussian_noise(x_test, severity="medium")   # create ONCE
_, acc_base = baseline.evaluate(x_noisy, y_test, verbose=0)
_, acc_aug = augmented.evaluate(x_noisy, y_test, verbose=0)
```
The augmentation layers inside `augmented.keras` are automatically inactive
during `evaluate()` / `predict()`, so both models see identical test images.

### Saved models
`.keras` files are **not** committed to Git (see `.gitignore`). To get them,
either run the training commands above (~5 min baseline, ~15 min augmented on a
laptop CPU) or get the files from Member 1. **Final reported results are
produced from Member 1's saved models.**

## Repository Structure
```
src/           core code (data, models, augmentations, corruptions, training, evaluation, plots)
scripts/       entry points for running experiments
results/       tables, figures, confusion matrices, training histories
saved_models/  trained models (not committed to Git)
docs/          report / presentation material
```

## Team Responsibilities
- **Nishant (ML / Training Lead):** data loading, CNN, augmentation, training — branch `feature/model-training`
- **Mahi (Corruption / Evaluation Lead):** corruptions, evaluation, plots — branch `feature/corruption-evaluation`

## Results So Far
Clean CIFAR-10 test accuracy (single training run per model):

| Model | Clean test accuracy | Best epoch | Stopped at |
|---|---:|---:|---:|
| Baseline CNN | 74.72% | 10 | 15 |
| Augmented CNN | 78.48% | 39 | 44 |

Corrupted-test results: _to be added after the evaluation pipeline is merged._

## Related Work
- Zhong et al., "Random Erasing Data Augmentation," AAAI 2020.
- Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," ICLR 2019.