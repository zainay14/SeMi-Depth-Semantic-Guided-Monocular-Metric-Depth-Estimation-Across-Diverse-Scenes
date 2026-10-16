# SeMi-Depth: Semantic-Guided Monocular Metric Depth Estimation Across Diverse Scenes

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Transactions-blue.svg)](Manuscript.pdf)

Official PyTorch implementation of **SeMi-Depth: Semantic-Guided Monocular Metric Depth Estimation Across Diverse Scenes**.

---

## 📌 Abstract
Monocular depth estimation (MDE) recovers a dense depth map from a single RGB image, yet predicting **metric** depth across heterogeneous environments remains difficult: indoor and outdoor scenes differ systematically in depth range ($0.3\,\text{m}-10\,\text{m}$ vs $0\,\text{m}-80\,\text{m}$), geometry, and illumination. Models trained on one domain learn priors that fail to transfer to the other.

**SeMi-Depth** is a unified multi-task framework that combines dense semantic guidance with explicit domain conditioning:
- A shared **Swin-Tiny** backbone extracts a multi-scale feature hierarchy.
- An autonomous **Scene Domain Classification Head** infers the operational domain (indoor vs. outdoor) and dynamically recalibrates activations via **Feature-wise Linear Modulation (FiLM)**.
- A **Semantic-Aware Cross-Attention Feature Modulation (CAF)** module explicitly aligns depth features with semantic boundary keys/values, eliminating depth bleed and silhouette halos.
- A **Dense Prediction Transformer (DPT)** decoder equipped with an **Adaptive Inverse-Depth Bins Head** outputs physically calibrated metric depth directly—**without requiring test-time ground-truth median scaling**.

---

## 🌟 Key Results

- **Cityscapes Outdoor Benchmark ($\le 50\,\text{m}$):**
  - $\text{AbsRel} = 0.1005$, $\text{SqRel} = 0.800$, $\text{RMSE} = 4.522\,\text{m}$, $\delta_1 = 89.1\%$.
  - Outperforms the $312\,\text{M}$-parameter foundation-model teacher of VFM-Depth by **$0.971\,\text{m}$ in RMSE** ($17.7\%$ error reduction) at only **$7\%$ of its computational cost**.
- **ScanNet Indoor Benchmark:**
  - $\text{RMSE} = 0.3466\,\text{m}$, $\delta_1 = 84.21\%$, rivaling heavy Swin-Large models with $8\times$ more parameters.
- **Inference Speed & Efficiency:**
  - **$28.3\,\text{M}$ parameters**, **$42.1\,\text{GFLOPs}$**, running at **$41.2\,\text{FPS}$** ($24.3\,\text{ms}$ latency) on a single NVIDIA RTX 4090 GPU.
- **Zero-Shot Cross-Dataset Transfer:**
  - Demonstrates robust generalization on **KITTI** (outdoor) and **NYU Depth V2** (indoor) without fine-tuning or scale assistance.

---

## 🏗️ Repository Architecture

```text
SeMi-Depth/
├── config.py                 # Central hyperparameter configuration
├── train.py                  # End-to-end multi-task joint training script
├── inference.py              # Single-image inference and visualization script
├── plot_curves.py            # Training & validation metric plotting utility
├── swa_average.py            # Stochastic Weight Averaging checkpoint utility
├── visualize_tsne.py         # t-SNE latent feature visualization
├── requirements.txt          # Python dependencies
├── .gitignore
├── models/
│   ├── backbone.py           # Hierarchical Swin-Tiny backbone
│   ├── depth_decoder.py      # DPT decoder with adaptive inverse-depth bins
│   ├── domain_head.py        # Scene domain classification head
│   ├── fusion.py             # Semantic Cross-Attention Modulation (CAF)
│   ├── seg_head.py           # Domain-conditioned semantic segmentation branch
│   └── full_model.py         # Complete unified multi-task SeMi-Depth model
├── data/
│   ├── dataset.py            # Joint domain-balanced dataloader (Cityscapes & ScanNet)
│   ├── prepare_dataset.py    # Preprocessing and dataset packaging
│   └── inspect_dataset.py    # Validation and statistical inspection utility
├── utils/
│   ├── losses.py             # Multi-task loss functions (distance-weighted, Chamfer, GCE)
│   ├── metrics.py            # Evaluation metrics (AbsRel, RMSE, delta accuracies, mIoU)
│   ├── transforms.py         # Geometric & photometric data augmentations
│   ├── tta.py                # Test-Time Augmentation (TTA) module
│   └── csv_logger.py         # Automated CSV training logger
└── demo_samples/             # Sample images for testing inference
    ├── outdoor_kitti_sample.jpg
    └── indoor_nyu_sample.png
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies in a Python 3.10+ environment:

```bash
git clone https://github.com/zainay14/SeMi-Depth-Semantic-Guided-Monocular-Metric-Depth-Estimation-Across-Diverse-Scenes.git
cd SeMi-Depth-Semantic-Guided-Monocular-Metric-Depth-Estimation-Across-Diverse-Scenes

pip install -r requirements.txt
```

### 2. Dataset Preparation
Download Cityscapes and ScanNet according to standard research protocols:
- **Cityscapes:** Place left RGB images, camera disparity/depth maps, and fine semantic annotations in `datasets/cityscapes`.
- **ScanNet:** Place extracted RGB frames, depth frames, and 20-class NYU-mapped semantics in `datasets/scannet`.

Preprocess the unified 15-class taxonomy:
```bash
python data/prepare_dataset.py --config config.py
```

### 3. Training
To train SeMi-Depth end-to-end with joint domain conditioning and multi-task supervision:
```bash
python train.py --config config.py
```

Checkpoints will be automatically saved under `checkpoints/` with automated metric evaluation logs.

### 4. Single-Image Inference
Run inference on an arbitrary input image (indoor or outdoor) to predict domain, 15-class semantics, and calibrated metric depth:
```bash
# Outdoor test sample
python inference.py --image demo_samples/outdoor_kitti_sample.jpg --ckpt checkpoints/best_model.pt --out ./outputs

# Indoor test sample
python inference.py --image demo_samples/indoor_nyu_sample.png --ckpt checkpoints/best_model.pt --out ./outputs
```

---

## 📖 Citation

If you find this code or paper useful in your research, please cite:

```bibtex
@article{iftikhar2026semidepth,
  author    = {Zainab Iftikhar},
  title     = {SeMi-Depth: Semantic-Guided Monocular Metric Depth Estimation Across Diverse Scenes},
  journal   = {IEEE Transactions},
  year      = {2026}
}
```

---

## 📜 License
This project is released under the [MIT License](LICENSE).
