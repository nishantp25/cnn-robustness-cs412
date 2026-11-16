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

## Related Work
- Zhong et al., "Random Erasing Data Augmentation," AAAI 2020.
- Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," ICLR 2019.

## Results
_To be filled in after experiments are run._