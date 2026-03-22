# DeepLense — Common Test I: Multi-Class Classification

**Google Summer of Code 2026 | ML4SCI | DeepLense**

---

## Task

> Build a model for classifying gravitational lensing images into three classes using PyTorch or Keras. Pick the most appropriate approach and discuss your strategy.

### Classes

| Label | Description |
|---|---|
| `no` | Strong lensing with **no substructure** |
| `sphere` | Strong lensing with **subhalo (spherical) substructure** |
| `vort` | Strong lensing with **vortex substructure** |

### Evaluation Metric
ROC curve (Receiver Operating Characteristic) and AUC score (Area Under the ROC Curve) — reported per class (One-vs-Rest) and as a macro average.

---

## Dataset

**Download:** [dataset.zip — Google Drive](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view)

After downloading, extract so the directory structure is:

```
dataset/
├── train/
│   ├── no/        # 10,000 images
│   ├── sphere/    # 10,000 images
│   └── vort/      # 10,000 images
└── val/
    ├── no/        #  2,500 images
    ├── sphere/    #  2,500 images
    └── vort/      #  2,500 images
```

Each image is a `.npy` file of shape `(1, 150, 150)`, dtype `float64`, already min-max normalised to `[0, 1]`.

**Split:** 90% train / 10% validation (30,000 train — 7,500 val), perfectly balanced across all three classes.

---

## Approach

### Model: ResNet18 with Transfer Learning

Pretrained ResNet18 (ImageNet) with the final classification head replaced to output 3 logits.

**Why ResNet18 over ResNet50?**
With 30k balanced 150×150 grayscale images across 3 classes, ResNet18 (11M parameters) provides ample capacity. ResNet50 (25M parameters) would overfit faster, train ~2× slower, and offer no meaningful gain for this task scale. ResNets are chosen over plain CNNs because their skip connections solve the vanishing gradient problem and enable effective feature reuse in deeper layers.

**Single-channel handling:**
The images are grayscale (`1` channel), but ResNet18 expects `3`-channel RGB input. Rather than reinitialising the first convolutional layer (losing pretrained weights), the single channel is repeated three times: `(1, 150, 150) → (3, 150, 150)`. This preserves the full ImageNet initialisation — the best possible starting point.

### Loss Function: CrossEntropyLoss

The dataset is perfectly balanced (10,000 samples per class), so Focal Loss provides no advantage here. Focal Loss is designed to down-weight easy examples via `(1 - p_t)^γ`, which is specifically beneficial when class imbalance causes many easy negatives to dominate the loss — not the case in this dataset.

### Optimiser: AdamW with Differential Learning Rates

| Parameter group | Learning rate |
|---|---|
| Pretrained backbone | `1e-4` (small — preserve ImageNet features) |
| New classification head | `1e-3` (larger — learn task quickly) |

**Why AdamW over Adam?** Plain Adam's weight decay is applied through the gradient update and interacts incorrectly with the adaptive learning rate. AdamW decouples weight decay from the gradient step, giving correct L2 regularisation.

### Scheduler: CosineAnnealingLR

Smoothly decays the learning rate from its initial value to near-zero following a cosine curve over 25 epochs. Avoids the abrupt LR drops of StepLR, which can destabilise fine-tuning.

### Data Augmentation

| Augmentation | Justification |
|---|---|
| Random horizontal flip | Gravitational lensing is reflectively symmetric |
| Random vertical flip | Same — no preferred orientation in the sky |
| Random rotation (±30°) | Rotationally symmetric physical process |

ColorJitter and strong crops are avoided — they could destroy the subtle substructure signal in the lensing ring.

---

## Results

Trained for 25 epochs on Apple M1 Pro (MPS backend) using AdamW + CosineAnnealingLR.

| Metric | Value |
|---|---|
| Validation Accuracy | **95.07%** |
| Macro AUC (OvR) | **0.9925** |
| AUC — No Substructure | 0.9933 |
| AUC — Subhalo Substructure | 0.9875 |
| AUC — Vortex Substructure | 0.9969 |

### Training Progression

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|---|---|---|---|---|---|
| 1 | 1.0235 | 45.31% | 0.8140 | 63.07% | 0.8303 |
| 5 | 0.3954 | 84.44% | 0.4272 | 85.21% | 0.9621 |
| 10 | 0.2827 | 89.38% | 0.2152 | 91.93% | 0.9846 |
| 15 | 0.2183 | 91.92% | 0.1767 | 93.80% | 0.9891 |
| 20 | 0.1729 | 93.48% | 0.1459 | 94.65% | 0.9920 |
| 25 | 0.1558 | 94.22% | 0.1405 | 95.05% | 0.9925 |

The best checkpoint (AUC = **0.9925**) was saved at epoch 24. The model converged cleanly with no signs of overfitting — train and val loss track closely throughout, and val AUC improved monotonically for the first 20 epochs before plateauing near the optimum.

---

## Notebook Walkthrough

[`solution.ipynb`](solution.ipynb) is structured as a single end-to-end pipeline:

| Section | Contents |
|---|---|
| **1. Setup** | Imports, device detection (CUDA → MPS → CPU), hyperparameters |
| **2. EDA** | Class distribution charts, sample images per class (viridis colormap), per-class pixel statistics, class-averaged images |
| **3. Dataset & DataLoaders** | Custom `LensDataset` for `.npy` files, train/val transforms, DataLoader setup |
| **4. Model** | ResNet18 pretrained, adapted for 3-class output, parameter count |
| **5. Training Loop** | Epoch-by-epoch table of loss, accuracy, and val AUC; best checkpoint saved by AUC |
| **6. Training Curves** | Loss / accuracy / AUC plots across epochs |
| **7. Evaluation** | ROC curves (all 3 classes on one plot), confusion matrix, classification report |
| **8. Grad-CAM** | Heatmap overlays showing which image regions the model attends to per class |
| **9. Summary** | Design choices and justification |

### Grad-CAM

Gradient-weighted Class Activation Mapping is used to visualise model attention. For each prediction, gradients of the predicted class score are backpropagated to the last convolutional layer (`layer4[1].conv2`), global-average-pooled into per-channel importance weights, and combined into a spatial heatmap. This confirms the model is attending to the **lensing ring and substructure region** rather than background noise — a critical sanity check for physics-informed ML.

---

## Setup

**1. Clone / download this repository**

**2. Download and extract the dataset** into a `dataset/` folder at the repo root (structure shown above).

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

> **Apple Silicon (M1/M2/M3/M4):** PyTorch will automatically use the MPS backend. Cell 1 will print `Using device: mps`. No extra configuration needed.

---

## Project Structure

```
Test-I/
├── dataset/               # Extracted dataset (not committed)
│   ├── train/
│   └── val/
├── solution.ipynb         # Main notebook — full pipeline
├── requirements.txt       # Python dependencies
└── README.md
```

---

## References

- He et al. (2015) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Loshchilov & Hutter (2019) — [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- Selvaraju et al. (2017) — [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- Lin et al. (2017) — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- ML4SCI DeepLense — [GSoC 2026 Test Instructions](https://ml4sci.org)
