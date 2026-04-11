import os
import torch
import random
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb

from models.multitask import MultiTaskPerceptionModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

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
    0: (0,   255, 0  ),
    1: (255, 0,   0  ),
    2: (255, 255, 0  ),
}


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
    image     = np.array(Image.open(image_path).convert('RGB'))
    transform = get_transform()
    tensor    = transform(image=image)['image']
    return tensor.unsqueeze(0), image


def postprocess_bbox(bbox_tensor, orig_h, orig_w):
    x_center, y_center, width, height = bbox_tensor[0].tolist()
    scale_x = orig_w / 224.0
    scale_y = orig_h / 224.0
    x_center *= scale_x
    y_center *= scale_y
    width    *= scale_x
    height   *= scale_y
    x1 = max(0, int(x_center - width  / 2))
    y1 = max(0, int(y_center - height / 2))
    x2 = min(orig_w, int(x_center + width  / 2))
    y2 = min(orig_h, int(y_center + height / 2))
    return x1, y1, x2, y2


def postprocess_segmentation(seg_tensor):
    pred_mask = seg_tensor[0].argmax(dim=0).cpu().numpy()
    colored   = np.zeros((224, 224, 3), dtype=np.uint8)
    for class_id, color in SEG_COLORS.items():
        colored[pred_mask == class_id] = color
    return pred_mask, colored


def run_inference(image_path, model):
    model.eval()
    tensor, orig_image = preprocess_image(image_path)
    tensor  = tensor.to(DEVICE)
    orig_h, orig_w = orig_image.shape[:2]

    with torch.no_grad():
        outputs = model(tensor)

    cls_out = outputs['classification']
    loc_out = outputs['localization']
    seg_out = outputs['segmentation']

    probs      = torch.softmax(cls_out, dim=1)[0]
    class_idx  = probs.argmax().item()
    confidence = probs[class_idx].item()
    class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"class_{class_idx}"

    x1, y1, x2, y2     = postprocess_bbox(loc_out.cpu(), orig_h, orig_w)
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


def visualize_and_save(results, image_name, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    orig  = results['orig_image']
    bbox  = results['bbox']
    mask  = results['colored_mask']
    label = results['class_name']
    conf  = results['confidence']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(orig)
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=3, edgecolor='red', facecolor='none'
    )
    axes[1].add_patch(rect)
    axes[1].set_title(f'Predicted: {label}\nConfidence: {conf:.2%}', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(mask)
    axes[2].set_title('Segmentation Mask', fontsize=12)
    axes[2].axis('off')

    legend_elements = [
        patches.Patch(facecolor='green',  label='Foreground'),
        patches.Patch(facecolor='red',    label='Background'),
        patches.Patch(facecolor='yellow', label='Uncertain'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.suptitle(f'Pipeline Output — {image_name}', fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{image_name}_result.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return save_path


def main():
    # ── W&B init ──
    wandb.init(
        project = 'da6401-assignment2',
        name    = 'pipeline-showcase-2.7'
    )

    # ── load model ──
    print("Loading MultiTaskPerceptionModel...")
    model = MultiTaskPerceptionModel(
        classifier_path = 'checkpoints/classifier.pth',
        localizer_path  = 'checkpoints/localizer.pth',
        unet_path       = 'checkpoints/unet.pth',
    ).to(DEVICE)

    # ── novel images ──
    novel_images = {
        'labrador'    : 'novel_pet1.jpg',
        'tabby_cat'   : 'novel_pet2.jpg',
        'mixed_breeds': 'novel_pet3.jpg',
    }

    # ── W&B table ──
    table = wandb.Table(columns=[
        'image_name', 'pipeline_output',
        'predicted_class', 'confidence',
        'bbox', 'generalization_notes'
    ])

    for name, path in novel_images.items():
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            continue

        print(f"\nRunning inference on: {path}")
        results   = run_inference(path, model)
        save_path = visualize_and_save(results, name)

        print(f"  Class      : {results['class_name']}")
        print(f"  Confidence : {results['confidence']:.2%}")
        print(f"  BBox       : {results['bbox']}")

        # generalization notes
        conf = results['confidence']
        if conf > 0.7:
            notes = "High confidence — model generalized well"
        elif conf > 0.4:
            notes = "Moderate confidence — partial generalization"
        else:
            notes = "Low confidence — out-of-distribution image"

        table.add_data(
            name,
            wandb.Image(save_path),
            results['class_name'],
            f"{conf:.2%}",
            str(results['bbox']),
            notes
        )

        # also log individual image
        wandb.log({f'pipeline_{name}': wandb.Image(save_path)})

    wandb.log({'pipeline_showcase': table})
    wandb.finish()
    print("\nDone! Results logged to W&B.")


if __name__ == '__main__':
    main()