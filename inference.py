import os
import torch
import random
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.multitask import MultiTaskPerceptionModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
    'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
    'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
    'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]

SEG_COLORS = {
    0: (0, 255, 0),    # foreground — green
    1: (255, 0, 0),    # background — red
    2: (255, 255, 0),  # uncertain  — yellow
}


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def get_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def preprocess_image(image_path):
    """Load and preprocess a single image for inference."""
    image     = np.array(Image.open(image_path).convert('RGB'))
    transform = get_transform()
    tensor    = transform(image=image)['image']
    return tensor.unsqueeze(0), image   # [1,3,224,224], original numpy


# ─────────────────────────────────────────────
# Postprocessing
# ─────────────────────────────────────────────

def postprocess_bbox(bbox_tensor, orig_h, orig_w):
    """Scale bbox from 224x224 space back to original image size."""
    x_center, y_center, width, height = bbox_tensor[0].tolist()

    scale_x = orig_w / 224.0
    scale_y = orig_h / 224.0

    x_center *= scale_x
    y_center *= scale_y
    width    *= scale_x
    height   *= scale_y

    x1 = int(x_center - width  / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width  / 2)
    y2 = int(y_center + height / 2)

    # clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(orig_w, x2)
    y2 = min(orig_h, y2)

    return x1, y1, x2, y2


def postprocess_segmentation(seg_tensor):
    """Convert segmentation logits to colored mask."""
    pred_mask = seg_tensor[0].argmax(dim=0).cpu().numpy()  # [224, 224]

    colored = np.zeros((224, 224, 3), dtype=np.uint8)
    for class_id, color in SEG_COLORS.items():
        colored[pred_mask == class_id] = color

    return pred_mask, colored


# ─────────────────────────────────────────────
# Main Inference
# ─────────────────────────────────────────────

def run_inference(image_path, model):
    model.eval()

    tensor, orig_image = preprocess_image(image_path)
    tensor = tensor.to(DEVICE)

    orig_h, orig_w = orig_image.shape[:2]

    # single forward pass only
    with torch.no_grad():
        cls_out, loc_out, seg_out = model(tensor)

    # classification
    probs      = torch.softmax(cls_out, dim=1)[0]
    class_idx  = probs.argmax().item()
    confidence = probs[class_idx].item()
    class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"class_{class_idx}"

    # localization
    x1, y1, x2, y2 = postprocess_bbox(loc_out.cpu(), orig_h, orig_w)

    # segmentation
    pred_mask, colored_mask = postprocess_segmentation(seg_out.cpu())

    return {
        'class_name'   : class_name,
        'class_idx'    : class_idx,
        'confidence'   : confidence,
        'bbox'         : (x1, y1, x2, y2),
        'seg_mask'     : pred_mask,
        'colored_mask' : colored_mask,
        'orig_image'   : orig_image,
    }
    

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def visualize_results(results, save_path=None):
    """Plot original image, bbox, and segmentation side by side."""
    orig   = results['orig_image']
    bbox   = results['bbox']
    mask   = results['colored_mask']
    label  = results['class_name']
    conf   = results['confidence']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # original image
    axes[0].imshow(orig)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # image with bounding box
    axes[1].imshow(orig)
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    axes[1].add_patch(rect)
    axes[1].set_title(f'{label} ({conf:.2%})')
    axes[1].axis('off')

    # segmentation mask
    axes[2].imshow(mask)
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')

    # legend for segmentation
    legend_elements = [
        patches.Patch(facecolor='green',  label='Foreground'),
        patches.Patch(facecolor='red',    label='Background'),
        patches.Patch(facecolor='yellow', label='Uncertain'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")

    plt.show()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description='DA6401 Assignment 2 Inference')
    parser.add_argument('--image',      type=str, required=True,
                        help='path to input image')
    parser.add_argument('--save',       type=str, default=None,
                        help='path to save visualization (optional)')
    parser.add_argument('--classifier', type=str, default='checkpoints/classifier.pth')
    parser.add_argument('--localizer',  type=str, default='checkpoints/localizer.pth')
    parser.add_argument('--unet',       type=str, default='checkpoints/unet.pth')
    args = parser.parse_args()

    print("Loading MultiTaskPerceptionModel...")
    model = MultiTaskPerceptionModel(
        classifier_path = args.classifier,
        localizer_path  = args.localizer,
        unet_path       = args.unet,
    ).to(DEVICE)

    print(f"Running inference on: {args.image}")
    results = run_inference(args.image, model)

    print(f"\n{'='*40}")
    print(f"Predicted Class : {results['class_name']}")
    print(f"Confidence      : {results['confidence']:.2%}")
    print(f"Bounding Box    : {results['bbox']}")
    print(f"{'='*40}\n")

    visualize_results(results, save_path=args.save)


if __name__ == '__main__':
    main()