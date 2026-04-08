import os
import argparse
import torch
import torch.nn as nn
import wandb
import numpy as np
from sklearn.metrics import f1_score

from data.pets_dataset import get_dataloaders
from models.classification import PetClassifier
from models.localization import PetLocalizer
from models.segmentation import PetSegmentor
from losses.iou_loss import IoULoss


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_f1(preds, labels):
    return f1_score(labels, preds, average='macro', zero_division=0)


def compute_iou_score(pred, target):
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    target_x1 = target[:, 0] - target[:, 2] / 2
    target_y1 = target[:, 1] - target[:, 3] / 2
    target_x2 = target[:, 0] + target[:, 2] / 2
    target_y2 = target[:, 1] + target[:, 3] / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w      = (inter_x2 - inter_x1).clamp(min=0)
    inter_h      = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h

    pred_area   = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)
    union       = pred_area + target_area - intersection

    return (intersection / (union + 1e-6)).mean().item()


def compute_dice(pred_mask, true_mask, num_classes=3):
    dice_scores = []
    pred_mask   = pred_mask.cpu().numpy()
    true_mask   = true_mask.cpu().numpy()

    for c in range(num_classes):
        pred_c        = (pred_mask == c).astype(np.float32)
        true_c        = (true_mask == c).astype(np.float32)
        intersection  = (pred_c * true_c).sum()
        union         = pred_c.sum() + true_c.sum()
        dice_scores.append(1.0 if union == 0 else 2 * intersection / union)

    return np.mean(dice_scores)


# ─────────────────────────────────────────────
# Task 1 — Classification
# ─────────────────────────────────────────────

def train_classifier(args):
    wandb.init(
        project = args.wandb_project,
        name    = 'task1_classification',
        config  = vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        root_dir    = args.data_dir,
        task        = 'classification',
        batch_size  = args.batch_size,
        num_workers = args.num_workers
    )
    # --- Compute class weights ---
    labels = [s['class_idx'] for s in train_loader.dataset.samples]
    counts = np.bincount(labels)

    weights = 1.0 / counts
    weights = weights / weights.sum()

    weights = torch.tensor(weights).float().to(device)
    
    model     = PetClassifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.05
    )

    # single lr for all params — model is trained from scratch, differential lr is only for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay
    )

    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_f1 = 0.0
    start_epoch = 0

    if args.resume_classifier:
        if os.path.exists(args.resume_classifier):
            ckpt = torch.load(args.resume_classifier, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            start_epoch = ckpt.get('epoch', 0)
            best_val_f1 = ckpt.get('val_f1', 0.0)

            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            else:
                # Backward-compatible resume for older checkpoints.
                for _ in range(start_epoch):
                    scheduler.step()

            print(f"Resumed classifier from {args.resume_classifier} at epoch {start_epoch} (best_val_f1={best_val_f1:.4f})")
        else:
            print(f"Resume checkpoint not found: {args.resume_classifier}. Starting from scratch.")

    no_improve = 0
    patience   = 10

    for epoch in range(start_epoch, args.epochs):

        # ── train ──
        model.train()
        train_loss             = 0.0
        all_preds, all_labels  = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_f1    = compute_f1(all_preds, all_labels)

        # ── val ──
        model.eval()
        val_loss               = 0.0
        val_preds, val_labels  = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs        = model(images)
                val_loss      += criterion(outputs, labels).item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_f1    = compute_f1(val_preds, val_labels)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Train F1: {train_f1:.4f} "
              f"Val Loss: {val_loss:.4f} Val F1: {val_f1:.4f}")

        wandb.log({
            'epoch'      : epoch + 1,
            'train/loss' : train_loss,
            'train/f1'   : train_f1,
            'val/loss'   : val_loss,
            'val/f1'     : val_f1,
            'lr'         : optimizer.param_groups[0]['lr']
        })

        # Always save latest state for seamless continuation after fixed epoch budgets.
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch'               : epoch + 1,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_f1'              : val_f1,
        }, 'checkpoints/classifier_last.pth')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
            torch.save({
                'epoch'               : epoch + 1,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1'              : val_f1,
            }, 'checkpoints/classifier.pth')
            print(f"  Saved best classifier (val_f1={val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    wandb.finish()
    print(f"Task 1 done. Best Val F1: {best_val_f1:.4f}")


# ─────────────────────────────────────────────
# Task 2 — Localization
# ─────────────────────────────────────────────

def train_localizer(args):
    wandb.init(
        project = args.wandb_project,
        name    = 'task2_localization',
        config  = vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, _ = get_dataloaders(
        root_dir    = args.data_dir,
        task        = 'localization',
        batch_size  = args.batch_size,
        num_workers = args.num_workers
    )

    model = PetLocalizer(
        dropout_p       = args.dropout_p,
        freeze_backbone = args.freeze_backbone
    ).to(device)

    if os.path.exists('checkpoints/classifier.pth'):
        model.load_backbone('checkpoints/classifier.pth')

    reg_loss  = nn.SmoothL1Loss()
    iou_loss  = IoULoss(reduction='mean')

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(),   'lr': args.lr * 0.1},
        {'params': model.regressor.parameters(),  'lr': args.lr}
    ], weight_decay=args.weight_decay)

    # 5-epoch linear warmup then cosine decay — warmup prevents gradient shock at epoch 1
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    best_val_iou = 0.0
    no_improve   = 0
    patience     = 10

    for epoch in range(args.epochs):

        # ── train ──
        model.train()
        train_loss = 0.0
        train_iou  = 0.0

        for images, bboxes in train_loader:
            images = images.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss  = reg_loss(preds, bboxes) + iou_loss(preds, bboxes)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_iou  += compute_iou_score(preds.detach(), bboxes)

        train_loss /= len(train_loader)
        train_iou  /= len(train_loader)

        # ── val ──
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0

        with torch.no_grad():
            for images, bboxes in val_loader:
                images   = images.to(device)
                bboxes   = bboxes.to(device)
                preds    = model(images)
                val_loss += (reg_loss(preds, bboxes) + iou_loss(preds, bboxes)).item()
                val_iou  += compute_iou_score(preds, bboxes)

        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Train IoU: {train_iou:.4f} "
              f"Val Loss: {val_loss:.4f} Val IoU: {val_iou:.4f}")

        wandb.log({
            'epoch'      : epoch + 1,
            'train/loss' : train_loss,
            'train/iou'  : train_iou,
            'val/loss'   : val_loss,
            'val/iou'    : val_iou,
            'lr'         : optimizer.param_groups[1]['lr']
        })

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improve   = 0
            torch.save({
                'epoch'               : epoch + 1,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou'             : val_iou,
            }, 'checkpoints/localizer.pth')
            print(f"  Saved best localizer (val_iou={val_iou:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    wandb.finish()
    print(f"Task 2 done. Best Val IoU: {best_val_iou:.4f}")


# ─────────────────────────────────────────────
# Task 3 — Segmentation
# ─────────────────────────────────────────────

def train_segmentor(args):
    wandb.init(
        project = args.wandb_project,
        name    = 'task3_segmentation',
        config  = vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, _ = get_dataloaders(
        root_dir    = args.data_dir,
        task        = 'segmentation',
        batch_size  = args.batch_size,
        num_workers = args.num_workers
    )

    model = PetSegmentor(
        num_classes     = 3,
        dropout_p       = args.dropout_p,
        freeze_backbone = args.freeze_backbone
    ).to(device)

    if os.path.exists('checkpoints/classifier.pth'):
        model.load_backbone('checkpoints/classifier.pth')

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = args.lr,
            weight_decay = args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_dice = 0.0
    no_improve    = 0
    patience      = 10

    for epoch in range(args.epochs):

        # ── train ──
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_dice += compute_dice(outputs.argmax(dim=1), masks)

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # ── val ──
        model.eval()
        val_loss      = 0.0
        val_dice      = 0.0
        val_pixel_acc = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images      = images.to(device)
                masks       = masks.to(device)
                outputs     = model(images)
                val_loss   += criterion(outputs, masks).item()
                pred_masks  = outputs.argmax(dim=1)
                val_dice   += compute_dice(pred_masks, masks)
                val_pixel_acc += (pred_masks == masks).float().mean().item()

        val_loss      /= len(val_loader)
        val_dice      /= len(val_loader)
        val_pixel_acc /= len(val_loader)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Train Dice: {train_dice:.4f} "
              f"Val Loss: {val_loss:.4f} Val Dice: {val_dice:.4f} "
              f"Val Pixel Acc: {val_pixel_acc:.4f}")

        wandb.log({
            'epoch'         : epoch + 1,
            'train/loss'    : train_loss,
            'train/dice'    : train_dice,
            'val/loss'      : val_loss,
            'val/dice'      : val_dice,
            'val/pixel_acc' : val_pixel_acc,
            'lr'            : optimizer.param_groups[0]['lr']
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            no_improve    = 0
            torch.save({
                'epoch'               : epoch + 1,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice'            : val_dice,
            }, 'checkpoints/unet.pth')
            print(f"  Saved best segmentor (val_dice={val_dice:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    wandb.finish()
    print(f"Task 3 done. Best Val Dice: {best_val_dice:.4f}")


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='DA6401 Assignment 2 Training')

    parser.add_argument('--task',            type=str,   default='classification',
                        choices=['classification', 'localization', 'segmentation'])
    parser.add_argument('--data_dir',        type=str,   default='./data/oxford-iiit-pet')
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--lr',              type=float, default=5e-4)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--dropout_p',       type=float, default=0.3)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--num_workers',     type=int,   default=2)
    parser.add_argument('--wandb_project',   type=str,   default='da6401-assignment2')
    parser.add_argument('--resume_classifier', type=str, default='')

    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    if args.task == 'classification':
        train_classifier(args)
    elif args.task == 'localization':
        train_localizer(args)
    elif args.task == 'segmentation':
        train_segmentor(args)