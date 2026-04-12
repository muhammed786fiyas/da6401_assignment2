# Checkpoints

Model checkpoints are NOT stored in this repository due to file size.
They are hosted on Google Drive and downloaded automatically at runtime.

## Automatic Download

Checkpoints are downloaded automatically when `MultiTaskPerceptionModel` 
is initialized via `gdown`. No manual download needed.

```python
from models.multitask import MultiTaskPerceptionModel

model = MultiTaskPerceptionModel(
    classifier_path = 'checkpoints/classifier.pth',
    localizer_path  = 'checkpoints/localizer.pth',
    unet_path       = 'checkpoints/unet.pth',
)
```

## Checkpoint Details

| File | Task | Val Metric | Epochs | Size |
|---|---|---|---|---|
| classifier.pth | Classification | F1 = 0.6503 | 50 | ~1350MB |
| localizer.pth  | Localization   | IoU = 0.6247 | 50 | ~520MB |
| unet.pth       | Segmentation   | Dice = 0.8454 | 50 | ~157MB |

## Google Drive IDs

| File | Drive ID |
|---|---|
| classifier.pth | 1aJ9_EkH8ZT1Eq_B5lL2cuw1GESTb75aI |
| localizer.pth  | 1WF1wcI_u2046AqMsHsDEGanNaXT9sTSm |
| unet.pth       | 1gGxfOgyoeANrvybKIDhHrovunbM1GW7Z |

## Training Details

All models trained on Google Colab / Kaggle with T4 GPU.
Dataset: Oxford-IIIT Pet Dataset
Framework: PyTorch 2.x
