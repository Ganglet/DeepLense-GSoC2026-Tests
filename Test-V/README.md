# DeepLense — Common Test V: Lens Finding & Data Pipelines

**Google Summer of Code 2026 | ML4SCI | DeepLense**

---

## Task

> Build a binary classification model to distinguish strong gravitational lenses from non-lensed galaxies. Demonstrate an effective data pipeline to handle extreme class imbalance.

### Classes

| Label | Description |
|---|---|
| `1` | Strong gravitational **lens** |
| `0` | **Non-lens** galaxy |

### Evaluation Metric
ROC curve (Receiver Operating Characteristic) and AUC score (Area Under the ROC Curve), plus Precision-Recall curve and confusion matrix at the Youden-optimal threshold.

---

## Dataset

50,055 `.npy` files, each `(3, 64, 64)` float32 — three observational filters (g, r, i) per object.

| Split | Lenses | Non-Lenses | Ratio |
|-------|--------|------------|-------|
| Train | 1,730  | 28,675     | 1 : 16.6 |
| Test  | 195    | 19,455     | 1 : 99.8 |

Dataset not included in this repo (2.5 GB). Place the `lens-finding-test/` directory at the repo root.

---

## Approach

### Model: ResNet18 with Transfer Learning

Pretrained ResNet18 (ImageNet) with the final FC replaced to output a single logit (binary head: `512 → 1`).

**Why ResNet18 over ResNet50?**
With 64×64 inputs and only 1,730 positive samples, ResNet18 (11M parameters) has ample capacity. ResNet50 (25M parameters) would overfit on the lens class, train ~2× slower, and offer no meaningful gain at this scale.

**3-channel input:**
The `.npy` arrays already provide three filters — no channel replication needed. ImageNet pretraining on 3-channel input maps naturally onto the (g, r, i) filter stack.

### Loss Function: Focal Loss

Focal Loss (γ=2, α=0.25) chosen over weighted BCE because of the extreme test-set skew (1:100). A fixed class weight in BCE doesn't adapt to prediction confidence — Focal Loss dynamically suppresses gradients from easy non-lenses the model already classifies correctly, pushing capacity toward hard lens examples.

### Optimiser: AdamW

| Parameter | Value |
|---|---|
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |

**Why AdamW over Adam?** AdamW decouples weight decay from the gradient step, giving correct L2 regularisation.

### Scheduler: CosineAnnealingLR

Avoids the abrupt LR drops of StepLR, which can destabilise fine-tuning. Paired with early stopping (patience=10) on validation AUC.

### Class Imbalance Strategy

`WeightedRandomSampler` ensures each batch is approximately 50% lenses. Without it, the first few epochs are dominated by non-lens gradients even with Focal Loss — the sampler gives the model enough positive examples per batch to learn useful lens features early.

### Data Augmentation

| Augmentation | Justification |
|---|---|
| Random horizontal flip | Gravitational lensing is reflectively symmetric |
| Random vertical flip | No preferred orientation in the sky |
| Random rotation (0–360°) | Einstein rings are fully rotationally symmetric |
| Gaussian noise (σ=0.01) | Regularises against pixel-level artefacts |
| Per-channel normalisation | Computed from training set; each filter has a distinct flux range |

No cropping — the Einstein ring structure spans the full 64×64 field, and cropping would destroy it.

---

## Results

Trained for up to 50 epochs on Apple M1 Pro (MPS backend) with early stopping. Best checkpoint saved at epoch 22 (Val AUC = **0.9937**); training stopped at epoch 32.

| Split | AUC |
|-------|-----|
| Validation (10% of train) | **0.9937** |
| Test | **0.9883** |

### Training Progression

| Epoch | Train Loss | Train AUC | Val Loss | Val AUC |
|---|---|---|---|---|
| 1  | 0.0304 | 0.9643 | 0.0232 | 0.9802 |
| 5  | 0.0182 | 0.9850 | 0.0230 | 0.9863 |
| 10 | 0.0123 | 0.9930 | 0.0132 | 0.9908 |
| 15 | 0.0096 | 0.9957 | 0.0128 | 0.9910 |
| 20 | 0.0081 | 0.9969 | 0.0119 | 0.9895 |
| 22 | 0.0078 | 0.9971 | 0.0082 | **0.9937** ← best |
| 32 | — | — | — | early stop |

The best checkpoint was saved at epoch 22. Val AUC plateaued after epoch 22 and early stopping triggered at epoch 32 (patience=10). Train loss continued to decrease smoothly, confirming the early stop correctly prevented overfitting.

---

## Notebook Walkthrough

[`solution.ipynb`](solution.ipynb) is structured as a single end-to-end pipeline:

| Section | Contents |
|---|---|
| **1. Setup** | Imports, device detection (CUDA → MPS → CPU), AMP flag, PRNG seeds |
| **2. Dataset Exploration** | File counts per split, shape/dtype/range verification, sample images per filter (g, r, i) for lens and non-lens |
| **3. Dataset Class** | Custom `LensDataset` for `.npy` arrays, lazy loading, float32 cast |
| **4. Normalisation & Augmentation** | Per-channel mean/std from training set (reservoir sampling), `PerChannelNormalize`, `AddGaussianNoise`, rotation/flip transforms |
| **5. Train/Val Split & DataLoaders** | 90:10 stratified split, `WeightedRandomSampler` for balanced batches, DataLoader setup |
| **6. Model** | ResNet18 pretrained, FC replaced with binary head, parameter count |
| **7. Focal Loss** | Custom `FocalLoss` module (α=0.25, γ=2.0), rationale vs weighted BCE |
| **8. Training** | Epoch loop with AMP support, early stopping on val AUC, best checkpoint saved |
| **9. Training Curves** | Loss and AUC plots across epochs (train vs val) |
| **10. Evaluation — Validation Set** | ROC + AUC, Precision-Recall curve, confusion matrix at Youden-optimal threshold |
| **11. Evaluation — Test Set** | Same evaluation suite on held-out test set |
| **12. Discussion** | Design choices, threshold strategy, potential improvements (TTA, EfficientNet-B0, self-supervised pretraining) |

---

## Setup

**1. Clone / download this repository**

**2. Download the dataset** and place `lens-finding-test/` at the repo root (structure shown in §Dataset above).

**3. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**4. Register the kernel with Jupyter**

```bash
python -m ipykernel install --user --name=gsoc-lens --display-name "GSoC DeepLense"
```

**5. Launch the notebook**

```bash
jupyter notebook solution.ipynb
```

Select the **GSoC DeepLense** kernel and run all cells top to bottom.

> **Apple Silicon (M1/M2/M3/M4):** PyTorch will automatically use the MPS backend. Cell 1 will print `Device: mps | AMP: False`. No extra configuration needed.

---

## Project Structure

```
Test-V/
├── lens-finding-test/         # Dataset (not committed)
│   ├── train_lenses/
│   ├── train_nonlenses/
│   ├── test_lenses/
│   └── test_nonlenses/
├── solution.ipynb             # Main notebook — full pipeline
├── requirements.txt           # Python dependencies
└── README.md
```

---

## References

- He et al. (2015) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Loshchilov & Hutter (2019) — [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- Lin et al. (2017) — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- ML4SCI DeepLense — [GSoC 2026 Test Instructions](https://ml4sci.org)
