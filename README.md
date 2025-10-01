# AIRL Internship Coding Assignment

This repository contains solutions for the AIRL Internship Coding Assignment, featuring implementations of two computer vision tasks:
- **Q1**: Vision Transformer (ViT) on CIFAR-10
- **Q2**: Text-Driven Image Segmentation with SAM 2

---

## 🚀 Quick Start

### Prerequisites
Both notebooks are designed to run in **Google Colab** with GPU acceleration enabled.

### Setup Instructions

1. **Enable GPU Runtime**
   - Navigate to [Google Colab](https://colab.research.google.com/)
   - Select `Runtime` → `Change runtime type` → Choose **T4 GPU** (or any available GPU)

2. **Clone Repository**
   ```bash
   !git clone https://github.com/realyashagarwal/airl-internship.git
   %cd airl-internship
   ```

3. **Run Notebooks**
   - Open `q1.ipynb` or `q2.ipynb` from the Colab file browser
   - Execute cells sequentially from top to bottom
   - Dependencies are automatically installed in the first code cell

---

## 📊 Question 1: Vision Transformer on CIFAR-10

### Objective
Implement a Vision Transformer (ViT) from scratch based on the paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) and achieve maximum test accuracy on CIFAR-10.

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `image_size` | 32 | Input image dimensions (CIFAR-10 default) |
| `patch_size` | 4 | Patch dimensions (4×4) |
| `embed_dim` | 384 | Embedding dimension per patch |
| `depth` | 12 | Number of Transformer encoder blocks |
| `num_heads` | 6 | Multi-Head Self-Attention heads |
| `mlp_ratio` | 4.0 | MLP hidden layer expansion ratio |
| `drop_path_rate` | 0.15 | Stochastic Depth rate |
| `epochs` | 200 | Total training epochs |
| `batch_size` | 128 | Training batch size |
| `base_lr` | 1e-3 | Base learning rate (AdamW) |
| `warmup_epochs` | 20 | Learning rate warmup period |
| `weight_decay` | 0.05 | AdamW weight decay |
| `augmentations` | CutMix, MixUp, Label Smoothing | Regularization techniques |

### Results

**Test Accuracy: 90.78%**

### Key Optimizations

#### Architecture
- **Convolutional Patch Embedding**: Efficient patchification using `nn.Conv2d` instead of linear projection
- **Pre-Layer Normalization**: Stabilizes training in deep Transformers by preventing gradient explosion
- **Layer Scale**: Learnable parameters control residual block contributions for improved training dynamics
- **Stochastic Depth (DropPath)**: Random block dropping acts as regularization, encouraging redundant feature learning

#### Data Augmentation
- **CutMix & MixUp**: Composite image training prevents overconfidence and improves generalization
- **Label Smoothing**: Softened labels discourage extreme logit predictions, acting as regularization

#### Training Strategy
- **AdamW Optimizer**: Decoupled weight decay yields better generalization than standard Adam
- **Warmup + Cosine Annealing**: Gradual LR warmup stabilizes early training; cosine decay aids convergence

---

## 🎯 Question 2: Text-Driven Image Segmentation with SAM 2

### Objective
Perform text-prompted segmentation on images using SAM 2, with bonus video object segmentation via temporal mask propagation.

### Pipeline Architecture

```
Text Prompt → GroundingDINO → Bounding Box → SAM 2 → Segmentation Mask
```

#### Implementation Steps

1. **Input**: Image + text prompt (e.g., "a dog")
2. **Text-to-Region**: GroundingDINO converts text to bounding box
3. **Region-to-Mask**: SAM 2 generates pixel-perfect mask from bounding box
4. **Visualization**: Mask overlaid on original image
5. **Bonus - Video**: Initial frame mask propagated temporally across video frames

### Limitations

| Limitation | Description |
|------------|-------------|
| **Detector Dependency** | Pipeline success entirely dependent on GroundingDINO's bounding box quality |
| **Text Understanding** | Struggles with abstract concepts, counting, and complex spatial relationships |
| **Video Tracking** | Object loss during occlusion, rapid motion, or minimal frame-to-frame movement |

---

## 📁 Repository Structure

```
├── q1.ipynb          # Vision Transformer implementation
├── q2.ipynb          # SAM 2 segmentation pipeline
└── README.md         # Documentation (this file)
```

---

## 📚 References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Segment Anything Model 2 (SAM 2)](https://ai.meta.com/sam2/)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

---
