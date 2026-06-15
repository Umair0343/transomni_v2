"""
Training Script for HybridScaleTransformer

Features:
1. Anti-Overfitting Techniques:
   - Class-balanced sampling
   - Learning rate warmup and cosine decay
   - Weight decay and dropout
   - MixUp augmentation
   - Early stopping

2. Multi-Scale Training:
   - Scale-aware loss weighting
   - Boundary-aware supervision
   - Deep supervision with auxiliary loss

3. Logging and Checkpointing:
   - TensorBoard logging
   - Best model saving
   - Metric tracking per class

Usage:
    python train.py --train_csv path/to/train.csv --val_csv path/to/val.csv --output_dir ./checkpoints
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional
import json
import csv
from datetime import datetime


class Logger:
    """
    Logger that writes to both terminal and file.

    Captures all print statements and saves them to a log file.
    """
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'a', buffering=1)  # Line buffered

        # Write header
        self.log_file.write("\n" + "=" * 80 + "\n")
        self.log_file.write(f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.write("\n" + "=" * 80 + "\n")
        self.log_file.write(f"Training Log - Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.close()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_cnn_transformer.models import build_hybrid_model, build_hats_model
from hybrid_cnn_transformer.losses import CombinedLoss, get_hats_loss, OnlineMixup, RenalAnatomyLoss, ConfusionPenaltyLoss
from hybrid_cnn_transformer.datasets import (
    PathologyDataset,
    PathologyValDataset,
    FullImageValDataset,
    ClassBalancedSampler,
    GroupedBatchSampler,
    collate_fn
)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """Compute Dice coefficient"""
    pred_binary = (torch.sigmoid(pred[:, 1]) > 0.5).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def compute_dice_binary(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """Compute Dice for binary prediction (already thresholded)"""
    pred_binary = pred.float()
    target_binary = target.float()
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    image: torch.Tensor,
    task_id: torch.Tensor,
    scale_id: torch.Tensor,
    patch_size: int = 512,
    overlap: float = 0.25,
    device: torch.device = None
) -> torch.Tensor:
    """
    Run sliding window inference on a full image.
    
    Args:
        model: The segmentation model
        image: Full image tensor [C, H, W]
        task_id: Task ID tensor
        scale_id: Scale ID tensor  
        patch_size: Size of patches to extract (default 512)
        overlap: Overlap ratio between patches (default 0.25)
        device: Device to run inference on
        
    Returns:
        Full-size binary prediction [H, W]
    """
    if device is None:
        device = next(model.parameters()).device
    
    C, H, W = image.shape
    stride = int(patch_size * (1 - overlap))
    
    # Initialize prediction accumulator and count
    pred_sum = torch.zeros((H, W), device=device, dtype=torch.float32)
    count = torch.zeros((H, W), device=device, dtype=torch.float32)
    
    # Slide window across image
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Extract patch
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patch_batch = patch.unsqueeze(0).to(device)  # [1, C, H, W]
            
            # Run model
            task_batch = task_id.unsqueeze(0).to(device)
            scale_batch = scale_id.unsqueeze(0).to(device)
            
            outputs = model(patch_batch, task_batch, scale_batch)
            
            # Get prediction probability
            pred_prob = torch.sigmoid(outputs['seg_logits'][:, 1])  # [1, H, W]
            
            # Accumulate
            pred_sum[y:y+patch_size, x:x+patch_size] += pred_prob.squeeze(0)
            count[y:y+patch_size, x:x+patch_size] += 1
    
    # Handle right and bottom edges if they weren't fully covered
    # Right edge
    if (W - patch_size) % stride != 0:
        x = W - patch_size
        for y in range(0, H - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patch_batch = patch.unsqueeze(0).to(device)
            task_batch = task_id.unsqueeze(0).to(device)
            scale_batch = scale_id.unsqueeze(0).to(device)
            outputs = model(patch_batch, task_batch, scale_batch)
            pred_prob = torch.sigmoid(outputs['seg_logits'][:, 1])
            pred_sum[y:y+patch_size, x:x+patch_size] += pred_prob.squeeze(0)
            count[y:y+patch_size, x:x+patch_size] += 1
    
    # Bottom edge
    if (H - patch_size) % stride != 0:
        y = H - patch_size
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patch_batch = patch.unsqueeze(0).to(device)
            task_batch = task_id.unsqueeze(0).to(device)
            scale_batch = scale_id.unsqueeze(0).to(device)
            outputs = model(patch_batch, task_batch, scale_batch)
            pred_prob = torch.sigmoid(outputs['seg_logits'][:, 1])
            pred_sum[y:y+patch_size, x:x+patch_size] += pred_prob.squeeze(0)
            count[y:y+patch_size, x:x+patch_size] += 1
    
    # Bottom-right corner
    if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
        y, x = H - patch_size, W - patch_size
        patch = image[:, y:y+patch_size, x:x+patch_size]
        patch_batch = patch.unsqueeze(0).to(device)
        task_batch = task_id.unsqueeze(0).to(device)
        scale_batch = scale_id.unsqueeze(0).to(device)
        outputs = model(patch_batch, task_batch, scale_batch)
        pred_prob = torch.sigmoid(outputs['seg_logits'][:, 1])
        pred_sum[y:y+patch_size, x:x+patch_size] += pred_prob.squeeze(0)
        count[y:y+patch_size, x:x+patch_size] += 1
    
    # Average predictions in overlap regions
    count = torch.clamp(count, min=1)  # Avoid division by zero
    pred_avg = pred_sum / count
    
    # Threshold to binary
    pred_binary = (pred_avg > 0.5).float()
    
    return pred_binary


def get_lr_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 5):
    """
    Get learning rate scheduler with warmup and cosine decay.

    Args:
        optimizer: Optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    active_classes: Optional[list] = None,
    mixup: Optional[nn.Module] = None,
    anatomy_loss_fn: Optional[nn.Module] = None,
    confusion_loss_fn: Optional[nn.Module] = None,
    loss_warmup_scale: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    if active_classes is None:
        active_classes = list(range(6))

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    anatomy_loss_meter = AverageMeter()
    confusion_loss_meter = AverageMeter()
    class_dice_meters = {i: AverageMeter() for i in active_classes}

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        weights = batch['weight'].to(device)
        task_ids = batch['task_id'].to(device)
        scale_ids = batch['scale_id'].to(device)

        # Apply MixUp augmentation if enabled (50% of batches)
        # Track whether this batch was mixed (for anatomy/confusion loss guard)
        batch_was_mixed = False
        if mixup is not None and np.random.random() < 0.5:
            images, labels, _ = mixup(images, labels)
            batch_was_mixed = True

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(images, task_ids, scale_ids)
            loss, loss_dict = criterion(outputs, labels, task_ids, scale_ids, weights)

            # === Anatomy Loss (skipped on mixed batches — needs clean binary masks) ===
            anatomy_loss = torch.tensor(0.0, device=device)
            if anatomy_loss_fn is not None and not batch_was_mixed:
                current_class = task_ids[0].item()
                related = anatomy_loss_fn.get_related_classes(current_class)

                if related:
                    # Get predictions for related classes (with no_grad for efficiency)
                    related_predictions = {}
                    with torch.no_grad():
                        for rel_class, _ in related:
                            if rel_class in active_classes:
                                rel_task_id = torch.full_like(task_ids, rel_class)
                                rel_outputs = model(images, rel_task_id, scale_ids)
                                rel_pred = torch.sigmoid(rel_outputs['seg_logits'][:, 1])
                                related_predictions[rel_class] = rel_pred

                    if related_predictions:
                        anatomy_loss, anat_dict = anatomy_loss_fn.compute_batch_loss(
                            labels.float(), related_predictions, current_class
                        )
                        # Apply warmup scale and weight
                        anatomy_loss = anatomy_loss * anatomy_loss_fn.weight * loss_warmup_scale

                loss = loss + anatomy_loss

            # === Confusion Penalty Loss (DT vs PT — skipped on mixed batches) ===
            confusion_loss = torch.tensor(0.0, device=device)
            if confusion_loss_fn is not None and not batch_was_mixed:
                current_class = task_ids[0].item()
                confused_class = confusion_loss_fn.get_confused_class(current_class)

                if confused_class is not None and confused_class in active_classes:
                    with torch.no_grad():
                        confused_task_id = torch.full_like(task_ids, confused_class)
                        confused_outputs = model(images, confused_task_id, scale_ids)
                        confused_pred = torch.sigmoid(confused_outputs['seg_logits'][:, 1])

                    confusion_loss = confusion_loss_fn(labels.float(), confused_pred)
                    # Apply warmup scale
                    confusion_loss = confusion_loss * loss_warmup_scale

                loss = loss + confusion_loss

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # Compute metrics
        with torch.no_grad():
            dice = compute_dice(outputs['seg_logits'], labels)

        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice, images.size(0))
        if anatomy_loss_fn is not None:
            anatomy_loss_meter.update(anatomy_loss.item(), images.size(0))
        if confusion_loss_fn is not None:
            confusion_loss_meter.update(confusion_loss.item(), images.size(0))

        # Track per-class dice
        task_id = task_ids[0].item()
        class_dice_meters[task_id].update(dice, images.size(0))

        # Print progress
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            anat_str = f" Anat: {anatomy_loss_meter.avg:.4f}" if anatomy_loss_fn else ""
            conf_str = f" Conf: {confusion_loss_meter.avg:.4f}" if confusion_loss_fn else ""
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss_meter.avg:.4f} Dice: {dice_meter.avg:.4f}{anat_str}{conf_str} "
                  f"Time: {elapsed:.1f}s")

    # Compile metrics
    metrics = {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg,
    }
    if anatomy_loss_fn is not None:
        metrics['anatomy_loss'] = anatomy_loss_meter.avg
    if confusion_loss_fn is not None:
        metrics['confusion_loss'] = confusion_loss_meter.avg
    for i in active_classes:
        if class_dice_meters[i].count > 0:
            metrics[f'dice_class{i}'] = class_dice_meters[i].avg

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    active_classes: Optional[list] = None
) -> Dict[str, float]:
    """Validate the model"""
    model.eval()

    if active_classes is None:
        active_classes = list(range(6))

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    class_dice_meters = {i: AverageMeter() for i in active_classes}

    for batch in val_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        task_ids = batch['task_id'].to(device)
        scale_ids = batch['scale_id'].to(device)

        outputs = model(images, task_ids, scale_ids)
        loss, _ = criterion(outputs, labels, task_ids, scale_ids)

        dice = compute_dice(outputs['seg_logits'], labels)

        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice, images.size(0))

        task_id = task_ids[0].item()
        class_dice_meters[task_id].update(dice, images.size(0))

    metrics = {
        'val_loss': loss_meter.avg,
        'val_dice': dice_meter.avg,
    }
    for i in active_classes:
        if class_dice_meters[i].count > 0:
            metrics[f'val_dice_class{i}'] = class_dice_meters[i].avg

    return metrics


@torch.no_grad()
def validate_with_tta(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    active_classes: Optional[list] = None,
    num_augmentations: int = 8
) -> Dict[str, float]:
    """
    Validate with Test-Time Augmentation (TTA).

    Applies flips and rotations, averages predictions for improved accuracy.
    Typically improves Dice by 1-3%.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device for inference
        active_classes: List of active class IDs
        num_augmentations: Number of TTA augmentations (4 or 8)
    """
    model.eval()

    if active_classes is None:
        active_classes = list(range(6))

    dice_meter = AverageMeter()
    class_dice_meters = {i: AverageMeter() for i in active_classes}

    # Define TTA transforms
    def get_tta_transforms(n=8):
        """Get TTA transforms and their inverses"""
        transforms = [
            (lambda x: x, lambda x: x),  # Original
            (lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-1])),  # H-flip
            (lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2])),  # V-flip
            (lambda x: torch.flip(x, [-1, -2]), lambda x: torch.flip(x, [-1, -2])),  # HV-flip
        ]
        if n == 8:
            transforms.extend([
                (lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, -1, [-2, -1])),  # 90°
                (lambda x: torch.rot90(x, 2, [-2, -1]), lambda x: torch.rot90(x, -2, [-2, -1])),  # 180°
                (lambda x: torch.rot90(x, 3, [-2, -1]), lambda x: torch.rot90(x, -3, [-2, -1])),  # 270°
                (lambda x: torch.flip(torch.rot90(x, 1, [-2, -1]), [-1]),
                 lambda x: torch.rot90(torch.flip(x, [-1]), -1, [-2, -1])),  # 90° + H-flip
            ])
        return transforms[:n]

    tta_transforms = get_tta_transforms(num_augmentations)

    for batch in val_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        task_ids = batch['task_id'].to(device)
        scale_ids = batch['scale_id'].to(device)

        # TTA: Average predictions over augmentations
        prob_sum = None
        for transform_fn, inverse_fn in tta_transforms:
            aug_images = transform_fn(images)
            outputs = model(aug_images, task_ids, scale_ids)
            probs = torch.sigmoid(outputs['seg_logits'][:, 1:2])
            probs_inv = inverse_fn(probs)

            if prob_sum is None:
                prob_sum = probs_inv
            else:
                prob_sum = prob_sum + probs_inv

        prob_avg = prob_sum / num_augmentations

        # Compute dice with averaged predictions
        preds = (prob_avg > 0.5).float().squeeze(1)
        dice = dice_coefficient(preds, labels)

        dice_meter.update(dice, images.size(0))
        task_id = task_ids[0].item()
        class_dice_meters[task_id].update(dice, images.size(0))

    metrics = {
        'val_dice_tta': dice_meter.avg,
    }
    for i in active_classes:
        if class_dice_meters[i].count > 0:
            metrics[f'val_dice_tta_class{i}'] = class_dice_meters[i].avg

    return metrics


def dice_coefficient(preds: torch.Tensor, labels: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient"""
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)
    intersection = (preds_flat * labels_flat).sum()
    return (2. * intersection + smooth) / (preds_flat.sum() + labels_flat.sum() + smooth)


@torch.no_grad()
def validate_sliding_window(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    active_classes: Optional[list] = None,
    patch_size: int = 512,
    overlap: float = 0.25
) -> Dict[str, float]:
    """
    Validate using sliding window inference on full images.
    
    This provides fair evaluation without using mask information
    for patch selection.
    
    Args:
        model: The segmentation model
        val_loader: DataLoader with FullImageValDataset
        device: Device to run on
        active_classes: List of active class IDs
        patch_size: Patch size for sliding window
        overlap: Overlap ratio between patches
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    if active_classes is None:
        active_classes = list(range(6))
    
    dice_meter = AverageMeter()
    class_dice_meters = {i: AverageMeter() for i in active_classes}
    
    for sample in val_loader:
        # Get full image and mask (batch size = 1)
        image = sample['image']  # [1, C, H, W] or [C, H, W]
        label = sample['label']  # [1, H, W] or [H, W]
        task_id = sample['task_id']
        scale_id = sample['scale_id']
        
        # Handle batch dimension
        if image.dim() == 4:
            image = image.squeeze(0)  # [C, H, W]
            label = label.squeeze(0)  # [H, W]
        if isinstance(task_id, torch.Tensor) and task_id.dim() > 0:
            task_id = task_id.squeeze(0)
            scale_id = scale_id.squeeze(0)
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)
        if not isinstance(task_id, torch.Tensor):
            task_id = torch.tensor(task_id)
        if not isinstance(scale_id, torch.Tensor):
            scale_id = torch.tensor(scale_id)
        
        _, H, W = image.shape
        
        # Skip images smaller than patch_size (use direct inference)
        if H < patch_size or W < patch_size:
            # Pad to patch_size
            pad_h = max(0, patch_size - H)
            pad_w = max(0, patch_size - W)
            image_padded = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
            label_padded = torch.nn.functional.pad(label, (0, pad_w, 0, pad_h))
            
            image_batch = image_padded.unsqueeze(0).to(device)
            task_batch = task_id.unsqueeze(0).to(device)
            scale_batch = scale_id.unsqueeze(0).to(device)
            
            outputs = model(image_batch, task_batch, scale_batch)
            pred = (torch.sigmoid(outputs['seg_logits'][:, 1]) > 0.5).float()
            
            # Crop back to original size
            pred = pred[:, :H, :W].squeeze(0)
            label_tensor = label_padded[:H, :W]
        else:
            # Run sliding window inference
            pred = sliding_window_inference(
                model, image, task_id, scale_id,
                patch_size=patch_size, overlap=overlap, device=device
            )
            label_tensor = label.to(device)
        
        # Compute Dice on full image
        dice = compute_dice_binary(pred, label_tensor)
        
        dice_meter.update(dice, 1)
        
        tid = task_id.item() if isinstance(task_id, torch.Tensor) else task_id
        if tid in class_dice_meters:
            class_dice_meters[tid].update(dice, 1)
    
    metrics = {
        'val_dice': dice_meter.avg,
    }
    for i in active_classes:
        if class_dice_meters[i].count > 0:
            metrics[f'val_dice_class{i}'] = class_dice_meters[i].avg
    
    return metrics


def convert_to_python_types(obj):
    """Convert PyTorch tensors and numpy arrays to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.item() if obj.size == 1 else obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': convert_to_python_types(metrics)
    }, path)


def train(args):
    """Main training function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging to file
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(args.output_dir, log_filename)
    logger = Logger(log_path)
    sys.stdout = logger
    print(f"Logging to: {log_path}")

    # Log training configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 60 + "\n")

    print(f"Using device: {device}")

    # Parse exclude_classes argument
    exclude_classes = None
    if args.exclude_classes:
        exclude_classes = [int(c.strip()) for c in args.exclude_classes.split(',')]
        print(f"\n*** Excluding classes: {exclude_classes} ***")

    # Define all class info
    ALL_CLASS_INFO = {
        0: 'DT', 1: 'PT', 2: 'CAP', 3: 'TUFT', 4: 'VES', 5: 'PTC'
    }

    # Compute active classes
    all_classes = set(range(6))
    excluded_set = set(exclude_classes) if exclude_classes else set()
    active_classes = sorted(all_classes - excluded_set)
    num_task_classes = len(active_classes)
    active_class_names = [ALL_CLASS_INFO[c] for c in active_classes]
    print(f"Active classes ({num_task_classes}): {active_class_names}")

    # Build model
    print("\nBuilding model...")
    print(f"Backbone: {args.backbone}")

    if args.backbone == 'hats':
        # HATs architecture with layer-wise token injection, transformer, and dynamic head
        # Enhanced version: 4 transformer layers (was 2), 16 head channels (was 8)
        print("Using Enhanced HATs architecture:")
        print("  - Layer-wise class/scale token injection")
        print("  - Transformer at bottleneck: 4 layers (enhanced from 2)")
        print("  - Dynamic head: 16 channels, 578 params (enhanced from 8ch, 162 params)")
        model = build_hats_model(
            num_classes=num_task_classes,
            num_scales=4,
            use_transformer=True,
            num_transformer_layers=4,  # Increased from 2
            head_channels=16,  # Increased from 8
            dropout=args.dropout
        )
    else:
        if args.backbone in ['resnet34', 'resnet50']:
            print(f"Using ImageNet pretrained: {args.pretrained_backbone}")
            print(f"Frozen stages: {args.freeze_backbone_stages}")

        model = build_hybrid_model(
            num_task_classes=num_task_classes,
            num_scales=4,
            use_transformer=args.use_transformer,
            backbone=args.backbone,
            pretrained_backbone=args.pretrained_backbone,
            freeze_backbone_stages=args.freeze_backbone_stages,
            dropout=args.dropout
        )

    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Count trainable vs total parameters (useful when using frozen backbone)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Build datasets
    print("\nLoading datasets...")
    print(f"Training CSV: {args.train_csv}")

    # Debug: Check CSV format
    try:
        import pandas as pd
        for sep in ['\t', ',', ';']:
            try:
                test_df = pd.read_csv(args.train_csv, sep=sep, nrows=2)
                print(f"CSV columns (sep='{repr(sep)}'): {list(test_df.columns)}")
                if len(test_df.columns) > 1:
                    break
            except:
                continue
    except Exception as e:
        print(f"Debug CSV check failed: {e}")

    train_dataset = PathologyDataset(
        csv_path=args.train_csv,
        crop_size=(512, 512),
        augment=True,
        edge_weight=True,
        max_iters=args.iters_per_epoch * args.batch_size,
        exclude_classes=exclude_classes
    )

    val_dataset = PathologyValDataset(
        csv_path=args.val_csv,
        target_size=(512, 512),  # Match training size for consistent evaluation
        exclude_classes=exclude_classes
    )
    
    # Create sliding window dataset if enabled
    if args.sliding_window_val:
        sliding_val_dataset = FullImageValDataset(
            csv_path=args.val_csv,
            max_size=2048,  # Max dimension to avoid OOM
            exclude_classes=exclude_classes
        )
        sliding_val_loader = DataLoader(
            sliding_val_dataset,
            batch_size=1,  # Full images, batch size must be 1
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print(f"Sliding window validation enabled (patch_size=512, overlap=25%)")

    # GroupedBatchSampler ensures all samples in a batch have the SAME class
    # This is required because task_id[0] is used to select tokens for the entire batch
    if args.class_balanced:
        sampler = GroupedBatchSampler(
            train_dataset, 
            batch_size=args.batch_size, 
            drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False  # GroupedBatchSampler handles drop_last internally
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Loss function
    if args.backbone == 'hats':
        # Simplified loss for HATs (CE + Dice, class-balanced)
        print("\nUsing simplified HATsLoss (CE + Dice)")
        criterion = get_hats_loss(use_class_balance=args.class_balanced, exclude_classes=exclude_classes)
    else:
        # Complex combined loss for other backbones
        criterion = CombinedLoss(
            dice_weight=1.0,
            focal_weight=0.5,
            boundary_weight=0.5,
            aux_weight=0.4,
            num_task_classes=num_task_classes,
            use_class_balance=args.class_balanced,
            exclude_classes=exclude_classes
        )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = get_lr_scheduler(optimizer, args.num_epochs, warmup_epochs=5)

    # Mixed precision
    scaler = GradScaler(enabled=args.use_amp)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

    # MixUp augmentation
    mixup = None
    if args.mixup:
        print(f"\nMixUp enabled with alpha={args.mixup_alpha}")
        mixup = OnlineMixup(alpha=args.mixup_alpha)

    # Renal Anatomy Loss (with warmup — inactive for first N epochs)
    anatomy_loss_fn = None
    anatomy_warmup_epoch = args.anatomy_warmup_epochs  # Epochs before activation
    if args.anatomy_loss:
        print(f"\nRenal Anatomy Loss enabled with weight={args.anatomy_loss_weight}")
        print(f"  Warmup: inactive for first {anatomy_warmup_epoch} epochs, then linear ramp-up over 10 epochs")
        print("  Relationships enforced:")
        print("    - TUFT must be inside CAP (subset)")
        print("    - CAP must contain TUFT (superset)")
        print("    - DT and PT are mutually exclusive")
        print("    - Tubules (DT, PT) exclusive with Glomeruli (CAP, TUFT)")
        anatomy_loss_fn = RenalAnatomyLoss(weight=args.anatomy_loss_weight)

    # DT vs PT Confusion Penalty Loss (same warmup as anatomy loss)
    confusion_loss_fn = None
    if args.confusion_loss:
        print(f"\nDT vs PT Confusion Penalty Loss enabled with weight={args.confusion_loss_weight}")
        print(f"  Warmup: inactive for first {anatomy_warmup_epoch} epochs (same as anatomy loss)")
        confusion_loss_fn = ConfusionPenaltyLoss(penalty_weight=args.confusion_loss_weight)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            if args.reset_lr:
                # Warm restart: keep model weights, but fresh optimizer and LR schedule
                print(f"*** LR RESET: Using fresh optimizer with lr={args.learning_rate}")
                print(f"*** Starting from epoch 0 with new LR schedule")
                # Don't load optimizer state, don't advance scheduler
                # start_epoch stays 0, best_dice stays 0
            else:
                # Normal resume: continue from where we left off
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                if 'metrics' in checkpoint and 'val_dice' in checkpoint['metrics']:
                    best_dice = checkpoint['metrics']['val_dice']
                print(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")
                # Advance scheduler to correct position
                for _ in range(start_epoch):
                    scheduler.step()
        else:
            print(f"\n*** WARNING: Checkpoint file not found: {args.resume}")
            print(f"*** Training will start from scratch!")
            print(f"*** Use --output_dir to specify where checkpoints are saved\n")

    # Training loop
    print("\nStarting training...")
    history = []

    try:
        for epoch in range(start_epoch, args.num_epochs):
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

            # Compute warmup scale for anatomy/confusion loss
            # Inactive for first anatomy_warmup_epoch epochs, then linear ramp over 10 epochs
            if epoch < anatomy_warmup_epoch:
                warmup_scale = 0.0
            elif epoch < anatomy_warmup_epoch + 10:
                warmup_scale = (epoch - anatomy_warmup_epoch + 1) / 10.0
            else:
                warmup_scale = 1.0

            # Only pass anatomy/confusion loss after warmup
            epoch_anatomy_fn = anatomy_loss_fn if warmup_scale > 0 else None
            epoch_confusion_fn = confusion_loss_fn if warmup_scale > 0 else None

            if warmup_scale > 0 and warmup_scale < 1.0:
                print(f"  Anatomy/Confusion loss warmup: {warmup_scale:.1%}")
            elif warmup_scale >= 1.0 and epoch == anatomy_warmup_epoch + 10:
                print("  Anatomy/Confusion loss: fully active")

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                device, epoch, args.use_amp, args.grad_clip, active_classes,
                mixup=mixup, anatomy_loss_fn=epoch_anatomy_fn,
                confusion_loss_fn=epoch_confusion_fn,
                loss_warmup_scale=warmup_scale
            )

            # Validate
            if (epoch + 1) % args.val_interval == 0:
                if args.sliding_window_val:
                    # Fair validation using sliding window (no mask guidance)
                    val_metrics = validate_sliding_window(
                        model, sliding_val_loader, device, active_classes,
                        patch_size=512, overlap=0.25
                    )
                    # No loss for sliding window validation
                    val_metrics['val_loss'] = 0.0
                elif args.use_tta:
                    # Validation with Test-Time Augmentation
                    val_metrics = validate_with_tta(
                        model, val_loader, criterion, device, active_classes,
                        num_augmentations=args.tta_augmentations
                    )
                    # Also run normal validation for loss
                    normal_val = validate(model, val_loader, criterion, device, active_classes)
                    val_metrics['val_loss'] = normal_val['val_loss']
                    # Use TTA dice as main metric
                    val_metrics['val_dice'] = val_metrics['val_dice_tta']
                    for i in active_classes:
                        tta_key = f'val_dice_tta_class{i}'
                        if tta_key in val_metrics:
                            val_metrics[f'val_dice_class{i}'] = val_metrics[tta_key]
                else:
                    # Patch-level validation (faster, uses mask guidance)
                    val_metrics = validate(model, val_loader, criterion, device, active_classes)

                metrics = {**train_metrics, **val_metrics}

                print(f"Train Loss: {train_metrics['loss']:.4f} Dice: {train_metrics['dice']:.4f}")
                if args.sliding_window_val:
                    print(f"Val Dice (sliding window): {val_metrics['val_dice']:.4f}")
                elif args.use_tta:
                    print(f"Val Loss: {val_metrics['val_loss']:.4f} Dice (TTA-{args.tta_augmentations}): {val_metrics['val_dice']:.4f}")
                else:
                    print(f"Val Loss: {val_metrics['val_loss']:.4f} Dice: {val_metrics['val_dice']:.4f}")

                # Print per-class metrics
                print("Per-class Dice:")
                for i, name in zip(active_classes, active_class_names):
                    train_key = f'dice_class{i}'
                    val_key = f'val_dice_class{i}'
                    if train_key in metrics:
                        print(f"  {name}: train={metrics.get(train_key, 0):.4f} "
                              f"val={metrics.get(val_key, 0):.4f}")

                # Save best model
                if val_metrics['val_dice'] > best_dice:
                    best_dice = val_metrics['val_dice']
                    save_checkpoint(
                        model, optimizer, epoch, metrics,
                        os.path.join(args.output_dir, 'best_model.pth')
                    )
                    print(f"New best model! Dice: {best_dice:.4f}")

                # Early stopping
                if early_stopping(val_metrics['val_dice']):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                metrics = train_metrics

            metrics['epoch'] = epoch + 1
            metrics['lr'] = scheduler.get_last_lr()[0]
            history.append(metrics)
            scheduler.step()

            # Write results after each epoch (overwrite so file always has latest)
            history_path = os.path.join(args.output_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(convert_to_python_types(history), f, indent=2)

            results_path = os.path.join(args.output_dir, 'epoch_results.txt')
            with open(results_path, 'w') as f:
                f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Dice':<12} {'Val Loss':<12} {'Val Dice':<12} {'Best':<6}")
                for name in active_class_names:
                    f.write(f" {name+'(T)':<10} {name+'(V)':<10}")
                f.write("\n" + "-" * (56 + 20 * len(active_class_names)) + "\n")
                for h in history:
                    ep = h.get('epoch', '?')
                    t_loss = h.get('loss', 0)
                    t_dice = h.get('dice', 0)
                    v_loss = h.get('val_loss', -1)
                    v_dice = h.get('val_dice', -1)
                    is_best = '*' if v_dice == best_dice and v_dice > 0 else ''
                    if v_loss >= 0:
                        f.write(f"{ep:<8} {t_loss:<12.4f} {t_dice:<12.4f} {v_loss:<12.4f} {v_dice:<12.4f} {is_best:<6}")
                    else:
                        f.write(f"{ep:<8} {t_loss:<12.4f} {t_dice:<12.4f} {'--':<12} {'--':<12} {'':<6}")
                    for i, name in zip(active_classes, active_class_names):
                        t_key = f'dice_class{i}'
                        v_key = f'val_dice_class{i}'
                        t_val = f"{h.get(t_key, 0):.4f}" if t_key in h else '--'
                        v_val = f"{h.get(v_key, 0):.4f}" if v_key in h else '--'
                        f.write(f" {t_val:<10} {v_val:<10}")
                    f.write("\n")
                f.write(f"\nBest Val Dice: {best_dice:.4f}\n")

            # Write per-epoch results to a CSV file (one row per epoch, always rewritten)
            csv_path = os.path.join(args.output_dir, 'metrics.csv')
            fieldnames = ['epoch', 'lr', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'is_best']
            for i, name in zip(active_classes, active_class_names):
                fieldnames.append(f'train_dice_{name}')
                fieldnames.append(f'val_dice_{name}')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for h in history:
                    v_dice = h.get('val_dice', None)
                    row = {
                        'epoch': h.get('epoch', ''),
                        'lr': f"{h.get('lr', float('nan')):.6f}" if 'lr' in h else '',
                        'train_loss': f"{h.get('loss', float('nan')):.6f}",
                        'train_dice': f"{h.get('dice', float('nan')):.6f}",
                        'val_loss': f"{h['val_loss']:.6f}" if 'val_loss' in h else '',
                        'val_dice': f"{v_dice:.6f}" if v_dice is not None else '',
                        'is_best': '1' if (v_dice is not None and v_dice == best_dice and v_dice > 0) else '',
                    }
                    for i, name in zip(active_classes, active_class_names):
                        t_key = f'dice_class{i}'
                        v_key = f'val_dice_class{i}'
                        row[f'train_dice_{name}'] = f"{h[t_key]:.6f}" if t_key in h else ''
                        row[f'val_dice_{name}'] = f"{h[v_key]:.6f}" if v_key in h else ''
                    writer.writerow(row)

            # Save latest model
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    model, optimizer, epoch, metrics,
                    os.path.join(args.output_dir, f'model_epoch{epoch+1}.pth')
                )

        print(f"\nTraining complete! Best Dice: {best_dice:.4f}")

    except KeyboardInterrupt:
        print(f"\n\n*** Training interrupted by user at epoch {epoch+1} ***")
        print(f"Best Dice so far: {best_dice:.4f}")
        # Save interrupted checkpoint
        save_checkpoint(
            model, optimizer, epoch, metrics if 'metrics' in dir() else train_metrics,
            os.path.join(args.output_dir, f'model_interrupted_epoch{epoch+1}.pth')
        )
        print(f"Interrupted checkpoint saved.")

    except Exception as e:
        print(f"\n\n*** Training failed with error: {e} ***")
        import traceback
        traceback.print_exc()

    finally:
        # Save training history (convert tensors to Python types for JSON serialization)
        if history:
            with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
                json.dump(convert_to_python_types(history), f, indent=2)
            print(f"Training history saved to: {os.path.join(args.output_dir, 'history.json')}")

        print(f"Log saved to: {log_path}")

        # Close logger and restore stdout
        sys.stdout = logger.terminal
        logger.close()
        print(f"Training log saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Train HybridScaleTransformer')

    # Data
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, required=True,
                        help='Path to validation CSV file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')

    # Model
    parser.add_argument('--use_transformer', action='store_true', default=True,
                        help='Use transformer encoder')
    parser.add_argument('--backbone', type=str, default='custom',
                        choices=['custom', 'resnet34', 'resnet50', 'hats'],
                        help='Backbone: custom, resnet34, resnet50, or hats (original HATs architecture)')
    parser.add_argument('--pretrained_backbone', action='store_true', default=True,
                        help='Use ImageNet pretrained backbone (only for resnet34/50)')
    parser.add_argument('--freeze_backbone_stages', type=int, default=0,
                        help='Number of backbone stages to freeze (0-4, only for resnet34/50)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')

    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--iters_per_epoch', type=int, default=250,
                        help='Iterations per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Regularization
    parser.add_argument('--class_balanced', action='store_true', default=True,
                        help='Use class-balanced sampling')
    parser.add_argument('--exclude_classes', type=str, default=None,
                        help='Comma-separated list of task_ids to exclude (e.g., "4,5" for VES,PTC)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='Use MixUp augmentation for regularization (blends samples)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='MixUp alpha parameter (controls blend strength)')
    parser.add_argument('--anatomy_loss', action='store_true', default=False,
                        help='Use Renal Anatomy Loss (enforces anatomical constraints between classes)')
    parser.add_argument('--anatomy_loss_weight', type=float, default=0.3,
                        help='Weight for anatomy loss (default 0.3)')
    parser.add_argument('--anatomy_warmup_epochs', type=int, default=60,
                        help='Number of epochs before anatomy/confusion loss activates (default 60)')
    parser.add_argument('--confusion_loss', action='store_true', default=False,
                        help='Use DT vs PT Confusion Penalty Loss (penalizes DT/PT misclassification)')
    parser.add_argument('--confusion_loss_weight', type=float, default=0.5,
                        help='Weight for confusion penalty loss (default 0.5)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    # Checkpointing
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validation interval (epochs)')
    parser.add_argument('--sliding_window_val', action='store_true', default=False,
                        help='Use sliding window inference for validation (fair, no mask guidance)')
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='Use Test-Time Augmentation during validation (improves Dice by 1-3%)')
    parser.add_argument('--tta_augmentations', type=int, default=8, choices=[4, 8],
                        help='Number of TTA augmentations (4=flips only, 8=flips+rotations)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint save interval (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--reset_lr', action='store_true', default=False,
                        help='Reset learning rate schedule when resuming (warm restart)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
