# DeepLense — Test V: Lens Finding & Data Pipelines

**GSOC 2026 | ML4Sci / DeepLense**

Binary classification of strong gravitational lenses vs non-lensed galaxies.

---

## Dataset

50,055 `.npy` files, each `(3, 64, 64)` float32 (three observational filters per object).

| Split | Lenses | Non-Lenses | Ratio |
|-------|--------|------------|-------|
| Train | 1,730  | 28,675     | 1 : 16.6 |
| Test  | 195    | 19,455     | 1 : 99.8 |

Dataset not included in this repo (2.5 GB). Place the `lens-finding-test/` directory at the repo root.

---

## Approach

**Model:** ResNet18 pretrained on ImageNet, final FC replaced with a 512→1 binary head.
Chose ResNet18 over deeper variants — 64×64 images and a 1,730-sample positive class don't justify the capacity of ResNet50/EfficientNet.

**Class imbalance:** `WeightedRandomSampler` for balanced batches + Focal Loss (γ=2, α=0.25).
Focal Loss handles the test-set skew (1:100) better than weighted BCE alone.

**Augmentation:** Random rotation (0–360°), horizontal/vertical flip, Gaussian noise, per-channel normalization.
No cropping — Einstein ring structure spans the full 64×64.

**Training:** AdamW + CosineAnnealingLR, early stopping on validation AUC.

**Evaluation:** ROC curve + AUC score (primary), Precision-Recall curve, confusion matrix.

---

## Results

| Split | AUC |
|-------|-----|
| Validation (10% of train) | — |
| Test | — |

*(Filled after training)*

---

## Setup

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

Open `solution.ipynb` and run all cells. The dataset path is configured in **Cell 2**.

---

## Structure

```
.
├── solution.ipynb       # Main notebook
├── README.md
└── .gitignore
```
