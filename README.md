# DA6401 Assignment 2 — Building a Complete Visual Perception Pipeline

## Links

| Resource | Link |
|---|---|
| W&B Report | [View Report](https://wandb.ai/muhammed786fiyas-iit-madras/da6401-assignment2/reports/DA6401-Assignment-2-Building-a-Complete-Visual-Perception-Pipeline--VmlldzoxNjQ1Njc2OA?accessToken=gekjkliedo31q13jh1iqhzmfzfvprzhabp5iocrf5xnn2ewt800n27vsjxb8g972) |
| GitHub Repo | [View Repository](https://github.com/muhammed786fiyas/da6401_assignment2) |

---

## Overview

End-to-end visual perception pipeline built on the **Oxford-IIIT Pet Dataset** using a VGG11 backbone implemented from scratch in PyTorch. The unified system performs three computer vision tasks simultaneously:

1. **Image Classification** — 37 pet breed classification (Val Macro F1: 0.6503)
2. **Object Localization** — Head bounding box regression (Val IoU: 0.6247)
3. **Semantic Segmentation** — Pixel-wise trimap prediction (Val Dice: 0.8454)

---
    
## Project Structure

```
da6401_assignment2/
├── checkpoints/
│   └── checkpoints.md
├── data/
│   └── pets_dataset.py
├── losses/
│   ├── __init__.py
│   └── iou_loss.py
├── models/
│   ├── __init__.py
│   ├── vgg11.py
│   ├── layers.py
│   ├── classification.py
│   ├── localization.py
│   ├── segmentation.py
│   └── multitask.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## Architecture

### VGG11 Backbone (implemented from scratch)
- 5 convolutional blocks following the original VGG11 paper
- BatchNorm2d injected after every Conv2d layer
- CustomDropout module (inherits nn.Module, inverted dropout scaling)
- Kaiming initialization for conv layers, Xavier uniform for linear layers

### Task 1 — Classification Head
```
Flatten → Linear(25088, 4096) → BatchNorm1d → ReLU → CustomDropout(p=0.3)
        → Linear(4096, 1024)  → BatchNorm1d → ReLU → CustomDropout(p=0.3)
        → Linear(1024, 37)
```

### Task 2 — Localization Head
```
Flatten → Linear(25088, 1024) → BatchNorm1d → ReLU → CustomDropout(p=0.3)
        → Linear(1024, 256)   → BatchNorm1d → ReLU → CustomDropout(p=0.3)
        → Linear(256, 4) → Clamp(0, 224)
Output: [x_center, y_center, width, height] in pixel space
```

### Task 3 — Segmentation (UNet Style)
```
Encoder: VGG11 split into 5 blocks → skip connections s1..s5
Decoder: 4 × DecoderBlock(ConvTranspose2d + skip concat + Conv2d)
Output:  [B, 3, 224, 224] — foreground / background / uncertain
```

### Unified MultiTask Pipeline
```
Input Image → VGG11Encoder (shared backbone)
                    ├── Classification Head → [B, 37]
                    ├── Localization Head   → [B, 4]
                    └── Segmentation Decoder → [B, 3, 224, 224]
```

---

## Setup

```bash
# clone repository
git clone https://github.com/muhammed786fiyas/da6401_assignment2.git
cd da6401_assignment2

# create conda environment
conda create -n da6401 python=3.10 -y
conda activate da6401

# install PyTorch (change cu118 to cpu if no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# install remaining dependencies
pip install -r requirements.txt
```

---

## Dataset Download

```python
import torchvision
torchvision.datasets.OxfordIIITPet(root='./data', download=True)
```

Expected structure after download:
```
data/oxford-iiit-pet/
├── images/
└── annotations/
    ├── list.txt
    ├── trimaps/
    └── xmls/
```

---

## Training

Train each task sequentially. Each task automatically loads the classifier backbone.

```bash
# Task 1 — Classification
python train.py \
    --task classification \
    --epochs 50 \
    --batch_size 64 \
    --lr 5e-4 \
    --dropout_p 0.3 \
    --num_workers 2 \
    --data_dir ./data/oxford-iiit-pet \
    --wandb_project da6401-assignment2

# Task 2 — Localization
python train.py \
    --task localization \
    --epochs 70 \
    --batch_size 64 \
    --lr 1e-3 \
    --dropout_p 0.3 \
    --num_workers 2 \
    --data_dir ./data/oxford-iiit-pet \
    --wandb_project da6401-assignment2

# Task 3 — Segmentation
python train.py \
    --task segmentation \
    --epochs 50 \
    --batch_size 32 \
    --lr 3e-4 \
    --dropout_p 0.3 \
    --num_workers 2 \
    --data_dir ./data/oxford-iiit-pet \
    --wandb_project da6401-assignment2
```

---

## Inference

```bash
python inference.py \
    --image path/to/your/pet.jpg \
    --classifier checkpoints/classifier.pth \
    --localizer  checkpoints/localizer.pth \
    --unet       checkpoints/unet.pth \
    --save       output.png
```

---

## Autograder Imports

The following imports are used by the autograder:

```python
from models.vgg11 import VGG11
from models.layers import CustomDropout
from losses.iou_loss import IoULoss
from models.multitask import MultiTaskPerceptionModel
```

---

## Results

### Validation Performance

| Task | Metric | Score |
|---|---|---|
| Classification | Val Macro F1 | 0.6503 |
| Localization | Val IoU | 0.6247 |
| Segmentation | Val Dice | 0.8454 |
| Segmentation | Val Pixel Accuracy | 0.9100 |

### Gradescope Results

| Test | Marks |
|---|---|
| VGG11 Architecture Verification | 5/5 |
| Custom Dropout Verification | 10/10 |
| Custom IoU Loss Verification | 5/5 |
| End-to-End Pipeline (Classification F1) | 10/10 |
| End-to-End Pipeline (Localization IoU) | 10/10 |
| End-to-End Pipeline (Segmentation Dice) | 10/10 |
| **Total** | **50/50** |

---

## Key Design Decisions

| Decision | Choice | Justification |
|---|---|---|
| BatchNorm placement | After every Conv2d and Linear | Enables 5x higher stable LR, 30% better F1 vs no BN |
| Dropout placement | Fully connected layers only | Preserves spatial feature maps for UNet skip connections |
| Dropout rate | p = 0.3 | Best balance between regularization and learning capacity |
| Localization loss | SmoothL1 + Custom IoU | SmoothL1 for coordinate stability, IoU for overlap quality |
| Segmentation loss | CrossEntropyLoss | Effective for 3-class moderate class imbalance |
| Transfer strategy | Full fine-tuning | +6.7% Dice improvement over frozen backbone |
| Upsampling method | ConvTranspose2d | Learnable upsampling, no bilinear interpolation |
| Bbox coordinate space | Pixel space [0, 224] | As required by assignment specification |

---

## Custom Components

### CustomDropout (models/layers.py)
```python
# Inverted dropout — scales by 1/(1-p) during training
# Deterministic passthrough during eval (self.training=False)
mask = (torch.rand(x.shape, device=x.device) > self.p).float()
return x * mask / (1 - self.p)
```

### IoULoss (losses/iou_loss.py)
```python
# Converts xywh → xyxy internally
# Loss = 1 - IoU, range [0, 1]
# Supports reduction: mean (default) and sum
```

### MultiTaskPerceptionModel (models/multitask.py)
```python
# Single forward pass returns dict
outputs = model(image)
outputs['classification']  # [B, 37]
outputs['localization']    # [B, 4]
outputs['segmentation']    # [B, 3, 224, 224]
```

---

## Requirements

```
torch
torchvision
numpy
matplotlib
scikit-learn
wandb
albumentations
gdown
```

---

## Notes

- Training was performed on Kaggle T4 GPU
- Dataset split: 70% train / 15% val / 15% test (fixed seed=42)
- All images normalized with ImageNet mean/std
- Checkpoints downloaded automatically via gdown at runtime
- No .pth files are stored in this repository

---

## Author

**Muhammed Fiyas**  
**DA25M018**       
IIT Madras