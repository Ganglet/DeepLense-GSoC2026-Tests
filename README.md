# DeepLense — GSoC 2026 Test Submissions

**Google Summer of Code 2026 | ML4SCI | DeepLense**

Gravitational lensing classification tasks submitted for the DeepLense GSoC 2026 evaluation.

---

## Tests

| Test | Task | Model | AUC |
|---|---|---|---|
| [Test I](Test-I/README.md) | Multi-class classification (3 lensing substructure types) | ResNet18 + transfer learning | **0.9925** macro |
| [Test V](Test-V/README.md) | Binary lens finding with extreme class imbalance (1:100) | ResNet18 + Focal Loss + WeightedRandomSampler | **0.9883** test |

---

## Test I — Multi-Class Classification

**Task:** Classify gravitational lensing images into three substructure types: no substructure, spherical subhalo, and vortex.

**Key choices:**
- ResNet18 pretrained on ImageNet; single-channel input replicated to 3 channels to preserve ImageNet initialisation
- CrossEntropyLoss — dataset is perfectly balanced (10k samples/class)
- AdamW with differential learning rates (backbone `1e-4`, head `1e-3`) + CosineAnnealingLR
- Augmentation: random flips + ±30° rotation (rotationally symmetric sky objects)

**Results:** Validation accuracy **95.07%**, macro AUC **0.9925** across 25 epochs on Apple M1 Pro.

→ [Full details](Test-I/README.md) · [Notebook](Test-I/solution.ipynb)

---

## Test V — Lens Finding & Data Pipelines

**Task:** Binary classification of strong gravitational lenses vs non-lensed galaxies under severe class imbalance (train 1:17, test 1:100).

**Key choices:**
- ResNet18 pretrained on ImageNet; 3-filter (g, r, i) `.npy` input maps directly onto the 3-channel conv layer
- Focal Loss (γ=2, α=0.25) — handles 1:100 test skew better than weighted BCE by dynamically suppressing easy-negative gradients
- `WeightedRandomSampler` for balanced batches + early stopping (patience=10) on val AUC
- Full 360° rotation augmentation — Einstein rings have no preferred orientation

**Results:** Validation AUC **0.9937**, test AUC **0.9883**. Best checkpoint at epoch 22; early stopping at epoch 32.

→ [Full details](Test-V/README.md) · [Notebook](Test-V/solution.ipynb)

---

## Setup

Each test is self-contained with its own virtual environment and `requirements.txt`. See the individual READMEs for setup instructions.

```
DeepLense-GSoC2026-Tests/
├── Test-I/          # Multi-class classification
│   ├── solution.ipynb
│   ├── requirements.txt
│   └── README.md
├── Test-V/         # Binary lens finding
│   ├── solution.ipynb
│   ├── requirements.txt
│   └── README.md
└── README.md        # This file
```

---

## References

- He et al. (2015) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Loshchilov & Hutter (2019) — [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- Lin et al. (2017) — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Selvaraju et al. (2017) — [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- ML4SCI DeepLense — [GSoC 2026 Test Instructions](https://ml4sci.org)
