import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, task='classification'):
        super(OxfordPetDataset, self).__init__()

        self.root_dir   = root_dir
        self.split      = split
        self.transform  = transform
        self.task       = task

        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir  = os.path.join(root_dir, 'annotations', 'trimaps')
        self.bbox_file  = os.path.join(root_dir, 'annotations', 'list.txt')

        self.samples     = []
        self.bbox_lookup = {}
        self._load_dataset()

    def _load_dataset(self):
        all_samples = []

        with open(self.bbox_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                image_name = parts[0]
                class_id   = int(parts[1]) - 1

                img_path  = os.path.join(self.images_dir, image_name + '.jpg')
                mask_path = os.path.join(self.masks_dir,  image_name + '.png')

                if not os.path.exists(img_path):
                    continue

                all_samples.append({
                    'image_name' : image_name,
                    'class_idx'  : class_id,
                    'img_path'   : img_path,
                    'mask_path'  : mask_path,
                })

        total   = len(all_samples)
        indices = list(range(total))
        rng     = np.random.default_rng(42)
        rng.shuffle(indices)

        train_end = int(0.70 * total)
        val_end   = int(0.85 * total)

        if self.split == 'train':
            selected = indices[:train_end]
        elif self.split == 'val':
            selected = indices[train_end:val_end]
        else:
            selected = indices[val_end:]

        self.samples     = [all_samples[i] for i in selected]
        self.bbox_lookup = self._load_bboxes()

    def _load_bboxes(self):
        import xml.etree.ElementTree as ET

        bbox_lookup = {}
        xml_dir     = os.path.join(self.root_dir, 'annotations', 'xmls')

        if not os.path.exists(xml_dir):
            return bbox_lookup

        for sample in self.samples:
            xml_path = os.path.join(xml_dir, sample['image_name'] + '.xml')
            if not os.path.exists(xml_path):
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj  = root.find('object')
            if obj is None:
                continue

            bndbox = obj.find('bndbox')
            xmin   = float(bndbox.find('xmin').text)
            ymin   = float(bndbox.find('ymin').text)
            xmax   = float(bndbox.find('xmax').text)
            ymax   = float(bndbox.find('ymax').text)

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width    = xmax - xmin
            height   = ymax - ymin

            bbox_lookup[sample['image_name']] = [x_center, y_center, width, height]

        return bbox_lookup

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image  = np.array(Image.open(sample['img_path']).convert('RGB'))

        mask = None
        if os.path.exists(sample['mask_path']):
            mask = np.array(Image.open(sample['mask_path']))
            mask = mask.astype(np.int64) - 1

        bbox = self.bbox_lookup.get(sample['image_name'], None)

        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image     = augmented['image']
                mask      = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image     = augmented['image']

        class_idx = torch.tensor(sample['class_idx'], dtype=torch.long)

        if self.task == 'classification':
            return image, class_idx

        elif self.task == 'localization':
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32) \
                          if bbox is not None else torch.zeros(4, dtype=torch.float32)
            return image, bbox_tensor

        elif self.task == 'segmentation':
            if mask is not None:
                return image, mask.long()
            return image, torch.zeros(224, 224, dtype=torch.long)

        elif self.task == 'multitask':
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32) \
                          if bbox is not None else torch.zeros(4, dtype=torch.float32)
            if mask is None:
                mask = torch.zeros(224, 224, dtype=torch.long)
            return image, class_idx, bbox_tensor, mask.long()

        return image, class_idx


def get_transforms(split='train', image_size=224):
    if split == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.Rotate(limit=20, p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2,
                rotate_limit=15, p=0.4
            ),
            A.HueSaturationValue(p=0.3),
            A.CoarseDropout(
                max_holes=8, max_height=32,
                max_width=32, p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


def get_dataloaders(root_dir, task='classification', batch_size=32, num_workers=4):
    train_dataset = OxfordPetDataset(
        root_dir=root_dir, split='train',
        transform=get_transforms('train'), task=task
    )
    val_dataset = OxfordPetDataset(
        root_dir=root_dir, split='val',
        transform=get_transforms('val'), task=task
    )
    test_dataset = OxfordPetDataset(
        root_dir=root_dir, split='test',
        transform=get_transforms('test'), task=task
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader