"""
Inference and Evaluation Utilities

Features:
1. Single image and batch prediction
2. Sliding window inference for large images
3. Metrics computation (Dice, F1, Precision, Recall)
4. Visualization utilities
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_cnn_transformer.models import build_hybrid_model


def load_model(
    checkpoint_path: str,
    device: torch.device = None,
    use_transformer: bool = True
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        use_transformer: Whether model uses transformer encoder

    Returns:
        Loaded model in evaluation mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_hybrid_model(
        num_task_classes=6,
        num_scales=4,
        use_transformer=use_transformer
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    return model


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    task_id: int,
    scale_id: int,
    device: torch.device = None,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Predict segmentation for a single image.

    Args:
        model: Trained model
        image: Input image [H, W, 3] or [3, H, W]
        task_id: Task/class identifier (0-5)
        scale_id: Scale identifier (0-3)
        device: Device for inference
        threshold: Binarization threshold

    Returns:
        Dictionary with:
            - 'mask': Binary segmentation mask [H, W]
            - 'prob': Probability map [H, W]
            - 'boundary': Boundary prediction [H, W]
    """
    if device is None:
        device = next(model.parameters()).device

    # Prepare input
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).float()
        if image.max() > 1:
            image = image / 255.0

    image = image.unsqueeze(0).to(device)  # Add batch dimension
    task_id_tensor = torch.tensor([task_id]).to(device)
    scale_id_tensor = torch.tensor([scale_id]).to(device)

    # Forward pass
    outputs = model(image, task_id_tensor, scale_id_tensor)

    # Process outputs
    seg_logits = outputs['seg_logits']
    prob = torch.sigmoid(seg_logits[:, 1])  # Foreground probability
    mask = (prob > threshold).float()

    # Get boundary if available
    boundary = None
    if 'boundary_logits' in outputs:
        boundary = torch.sigmoid(outputs['boundary_logits'][:, 0])

    # Convert to numpy
    results = {
        'mask': mask[0].cpu().numpy(),
        'prob': prob[0].cpu().numpy()
    }
    if boundary is not None:
        results['boundary'] = boundary[0].cpu().numpy()

    return results


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    task_id: int,
    scale_id: int,
    device: torch.device = None,
    threshold: float = 0.5,
    num_augmentations: int = 8
) -> Dict[str, np.ndarray]:
    """
    Predict with Test-Time Augmentation (TTA) for improved accuracy.

    Applies multiple augmentations (flips, rotations) and averages predictions.
    Can improve Dice by 2-5% without retraining!

    Args:
        model: Trained model
        image: Input image [H, W, 3] or [3, H, W]
        task_id: Task/class identifier (0-5)
        scale_id: Scale identifier (0-3)
        device: Device for inference
        threshold: Binarization threshold
        num_augmentations: Number of augmentations (4 or 8)

    Returns:
        Dictionary with averaged predictions
    """
    if device is None:
        device = next(model.parameters()).device

    # Prepare input
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).float()
        if image.max() > 1:
            image = image / 255.0

    image = image.unsqueeze(0).to(device)
    task_id_tensor = torch.tensor([task_id]).to(device)
    scale_id_tensor = torch.tensor([scale_id]).to(device)

    # Define augmentations and their inverses
    augmentations = [
        # (transform_fn, inverse_fn)
        (lambda x: x, lambda x: x),  # Original
        (lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-1])),  # H-flip
        (lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2])),  # V-flip
        (lambda x: torch.flip(x, [-1, -2]), lambda x: torch.flip(x, [-1, -2])),  # HV-flip
    ]

    if num_augmentations == 8:
        # Add 90-degree rotations
        augmentations.extend([
            (lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, -1, [-2, -1])),  # 90°
            (lambda x: torch.rot90(x, 2, [-2, -1]), lambda x: torch.rot90(x, -2, [-2, -1])),  # 180°
            (lambda x: torch.rot90(x, 3, [-2, -1]), lambda x: torch.rot90(x, -3, [-2, -1])),  # 270°
            (lambda x: torch.flip(torch.rot90(x, 1, [-2, -1]), [-1]),
             lambda x: torch.rot90(torch.flip(x, [-1]), -1, [-2, -1])),  # 90° + H-flip
        ])

    # Collect predictions
    prob_sum = None

    for transform_fn, inverse_fn in augmentations[:num_augmentations]:
        # Apply augmentation
        aug_image = transform_fn(image)

        # Forward pass
        outputs = model(aug_image, task_id_tensor, scale_id_tensor)
        seg_logits = outputs['seg_logits']
        prob = torch.sigmoid(seg_logits[:, 1:2])  # Keep dim for inverse

        # Inverse augmentation on prediction
        prob_inv = inverse_fn(prob)

        # Accumulate
        if prob_sum is None:
            prob_sum = prob_inv
        else:
            prob_sum = prob_sum + prob_inv

    # Average predictions
    prob_avg = prob_sum / num_augmentations
    mask = (prob_avg > threshold).float()

    return {
        'mask': mask[0, 0].cpu().numpy(),
        'prob': prob_avg[0, 0].cpu().numpy()
    }


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Predict segmentation for a batch of images.

    Args:
        model: Trained model
        dataloader: DataLoader with images
        device: Device for inference
        threshold: Binarization threshold

    Returns:
        List of prediction dictionaries
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    for batch in dataloader:
        images = batch['image'].to(device)
        task_ids = batch['task_id'].to(device)
        scale_ids = batch['scale_id'].to(device)
        names = batch['name']

        outputs = model(images, task_ids, scale_ids)

        prob = torch.sigmoid(outputs['seg_logits'][:, 1])
        masks = (prob > threshold).float()

        for i in range(len(names)):
            results.append({
                'name': names[i],
                'mask': masks[i].cpu().numpy(),
                'prob': prob[i].cpu().numpy(),
                'task_id': task_ids[i].item(),
                'scale_id': scale_ids[i].item()
            })

    return results


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    task_id: int,
    scale_id: int,
    window_size: Tuple[int, int] = (512, 512),
    stride: Tuple[int, int] = (256, 256),
    device: torch.device = None,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Perform sliding window inference for large images.

    Useful for 3000x3000 panoramic pathology images.

    Args:
        model: Trained model
        image: Large input image [H, W, 3]
        task_id: Task identifier
        scale_id: Scale identifier
        window_size: Size of sliding window
        stride: Stride between windows
        device: Device for inference
        threshold: Binarization threshold

    Returns:
        Full resolution segmentation
    """
    if device is None:
        device = next(model.parameters()).device

    H, W = image.shape[:2]
    win_h, win_w = window_size
    stride_h, stride_w = stride

    # Initialize output arrays
    prob_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    # Compute padding if needed
    pad_h = (win_h - H % stride_h) % stride_h if H % stride_h != 0 else 0
    pad_w = (win_w - W % stride_w) % stride_w if W % stride_w != 0 else 0

    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    # Slide over image
    for y in range(0, image.shape[0] - win_h + 1, stride_h):
        for x in range(0, image.shape[1] - win_w + 1, stride_w):
            # Extract patch
            patch = image[y:y+win_h, x:x+win_w]

            # Predict
            result = predict_single(model, patch, task_id, scale_id, device, threshold=0.0)
            prob = result['prob']

            # Accumulate (handle boundary)
            y_end = min(y + win_h, H)
            x_end = min(x + win_w, H)
            prob_h = y_end - y
            prob_w = x_end - x

            prob_sum[y:y_end, x:x_end] += prob[:prob_h, :prob_w]
            count[y:y_end, x:x_end] += 1

    # Average overlapping regions
    prob_avg = prob_sum / np.maximum(count, 1)
    mask = (prob_avg > threshold).astype(np.float32)

    return {
        'mask': mask,
        'prob': prob_avg
    }


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Compute segmentation metrics.

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dictionary with Dice, F1, Precision, Recall, IoU
    """
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)

    tp = np.sum(pred & target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    tn = np.sum(~pred & ~target)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'dice': dice,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'accuracy': accuracy
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: Validation/test DataLoader
        device: Device for inference
        threshold: Binarization threshold

    Returns:
        Dictionary of metrics (overall and per-class)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_metrics = []
    class_metrics = {i: [] for i in range(6)}
    class_names = ['DT', 'PT', 'CAP', 'TUFT', 'VES', 'PTC']

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        task_ids = batch['task_id'].numpy()
        scale_ids = batch['scale_id'].numpy()

        with torch.no_grad():
            outputs = model(images, batch['task_id'].to(device), batch['scale_id'].to(device))
            prob = torch.sigmoid(outputs['seg_logits'][:, 1])
            masks = (prob > threshold).float().cpu().numpy()

        for i in range(len(labels)):
            metrics = compute_metrics(masks[i], labels[i])
            all_metrics.append(metrics)
            class_metrics[task_ids[i]].append(metrics)

    # Aggregate metrics
    results = {}

    # Overall metrics
    for key in all_metrics[0].keys():
        results[key] = np.mean([m[key] for m in all_metrics])

    # Per-class metrics
    for class_id, metrics_list in class_metrics.items():
        if metrics_list:
            for key in metrics_list[0].keys():
                results[f'{class_names[class_id]}_{key}'] = np.mean([m[key] for m in metrics_list])

    return results


def visualize_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    save_path: Optional[str] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Visualize prediction overlay on image.

    Args:
        image: Original image [H, W, 3]
        mask: Ground truth mask [H, W]
        pred: Predicted mask [H, W]
        save_path: Optional path to save visualization
        alpha: Overlay transparency

    Returns:
        Visualization image
    """
    # Ensure image is RGB
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    vis = image.copy()

    # Create colored overlays
    # Green for true positive, Red for false positive, Blue for false negative
    tp_mask = (pred > 0.5) & (mask > 0.5)
    fp_mask = (pred > 0.5) & (mask < 0.5)
    fn_mask = (pred < 0.5) & (mask > 0.5)

    # Apply overlays
    vis[tp_mask] = (1 - alpha) * vis[tp_mask] + alpha * np.array([0, 255, 0])  # Green
    vis[fp_mask] = (1 - alpha) * vis[fp_mask] + alpha * np.array([255, 0, 0])  # Red
    vis[fn_mask] = (1 - alpha) * vis[fn_mask] + alpha * np.array([0, 0, 255])  # Blue

    vis = vis.astype(np.uint8)

    if save_path:
        Image.fromarray(vis).save(save_path)

    return vis


if __name__ == '__main__':
    # Test inference utilities
    print("Testing inference utilities...")

    # Create dummy model
    model = build_hybrid_model(use_transformer=True)
    model.eval()

    # Test single prediction
    dummy_image = np.random.rand(512, 512, 3).astype(np.float32)
    result = predict_single(model, dummy_image, task_id=0, scale_id=1)

    print(f"Single prediction:")
    print(f"  Mask shape: {result['mask'].shape}")
    print(f"  Prob shape: {result['prob'].shape}")

    # Test metrics
    pred = np.random.rand(512, 512) > 0.5
    target = np.random.rand(512, 512) > 0.5
    metrics = compute_metrics(pred, target)

    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nInference utilities test passed!")
