"""
Visualize model predictions for TUFT and CAP classes.

Color coding:
- Blue: True Positive (correctly predicted)
- Yellow: False Positive (predicted but not in ground truth)
- Magenta: False Negative (in ground truth but not predicted)
- Original image visible underneath

Usage:
 python visualize_predictions.py \
 --image path/to/image.png \
 --mask path/to/ground_truth_mask.png \
 --model path/to/best_model.pth \
 --task_id 2 # 2=CAP, 3=TUFT \
 --output visualization.png
"""

import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import sys

# Add the project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# If running from inside hybrid_cnn_transformer, add parent to path
if os.path.basename(script_dir) == 'hybrid_cnn_transformer':
    sys.path.insert(0, parent_dir)
else:
    sys.path.insert(0, script_dir)

from hybrid_cnn_transformer.models.hybrid_scale_transformer import HybridScaleTransformer


def load_model(checkpoint_path, device='cuda'):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Initialize model
    model = HybridScaleTransformer(
        num_classes=2,  # 6 classes
        num_scales=4,   # 4 scales
        backbone='custom',
        use_transformer=True,
        dropout=0.2
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
    return model


def load_image_and_mask(image_path, mask_path):
    """Load image and ground truth mask."""
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))

    # Load mask
    mask = np.array(Image.open(mask_path).convert('L'))

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    return image, mask


def predict_patch(model, image_patch, task_id, scale_id, device):
    # Normalize image
    image_tensor = torch.from_numpy(image_patch).float().permute(2,0,1)/255.0
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    image_tensor = (image_tensor - mean)/std
    image_tensor = image_tensor.unsqueeze(0).to(device)

    task_token = torch.tensor([task_id], dtype=torch.long, device=device)
    scale_token = torch.tensor([scale_id], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(image_tensor, task_token, scale_token)
        # Select the foreground logits for the binary task
        logits = outputs['seg_logits'][:, 1, :, :]  # [1, H, W]
        prob = torch.sigmoid(logits)
        prediction = (prob > 0.5).float().squeeze().cpu().numpy()

    return prediction


def predict_full_image(model, image, task_id, scale_id, patch_size=512, device='cuda'):
    """
    Run sliding window inference on full image.

    Args:
        model: Trained model
        image: [H, W, 3] numpy array
        task_id: Class to predict
        scale_id: Magnification scale
        patch_size: Patch size for sliding window
        device: torch device

    Returns:
        prediction: [H, W] binary mask
    """
    h, w = image.shape[:2]

    # If image smaller than patch_size, pad it
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        prediction_padded = predict_patch(model, image_padded, task_id, scale_id, device)
        return prediction_padded[:h, :w]

    # If image larger, use sliding window
    stride = patch_size // 2
    prediction_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    print(f"Running sliding window inference (stride={stride})...")

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            pred_patch = predict_patch(model, patch, task_id, scale_id, device)

            prediction_map[y:y+patch_size, x:x+patch_size] += pred_patch
            count_map[y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping predictions
    prediction_map = prediction_map / np.maximum(count_map, 1)
    prediction = (prediction_map > 0.5).astype(np.uint8)

    return prediction


def create_overlay_visualization(image, ground_truth, prediction, alpha=0.5):
    """
    Create visualization overlay.

    Color coding:
    - Blue: True Positive (TP) - predicted and in ground truth
    - Red: False Positive (FP) - predicted but not in ground truth
    - Green: False Negative (FN) - in ground truth but not predicted

    Args:
        image: [H, W, 3] original image
        ground_truth: [H, W] binary mask (0 or 255)
        prediction: [H, W] binary mask (0 or 1)
        alpha: Transparency of overlay

    Returns:
        overlay: [H, W, 3] visualization
    """
    # Normalize masks to binary
    gt_binary = (ground_truth > 127).astype(np.uint8)
    pred_binary = prediction.astype(np.uint8)

    # Calculate TP, FP, FN
    tp = (pred_binary == 1) & (gt_binary == 1)
    fp = (pred_binary == 1) & (gt_binary == 0)
    fn = (pred_binary == 0) & (gt_binary == 1)

    # Create colored overlay
    overlay = image.copy().astype(np.float32)

    # Blue for TP (correct predictions)
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 100, 255]) * alpha

    # Yellow for FP (incorrect predictions)
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([255, 255, 0]) * alpha

    # Magenta for FN (missed ground truth)
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([255, 0, 255]) * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Calculate metrics
    tp_count = tp.sum()
    fp_count = fp.sum()
    fn_count = fn.sum()

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0

    metrics = {
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'precision': precision,
        'recall': recall,
        'dice': dice
    }

    return overlay, metrics


def visualize(image_path, mask_path, model_path, task_id, scale_id=0, output_path=None, device='cuda'):
    """
    Main visualization function.

    Args:
        image_path: Path to input image
        mask_path: Path to ground truth mask
        model_path: Path to trained model checkpoint
        task_id: 2=CAP, 3=TUFT
        scale_id: 0=5x, 1=10x, 2=20x, 3=40x
        output_path: Path to save visualization (optional)
        device: 'cuda' or 'cpu'
    """
    # Class names
    class_names = {0: 'DT', 1: 'PT', 2: 'CAP', 3: 'TUFT', 4: 'VES', 5: 'PTC'}
    class_name = class_names[task_id]

    print(f"\n{'='*60}")
    print(f"Visualizing {class_name} predictions")
    print(f"{'='*60}")

    # Load model
    model = load_model(model_path, device)

    # Load image and mask
    image, ground_truth = load_image_and_mask(image_path, mask_path)

    # Run prediction
    print("Running inference...")
    prediction = predict_full_image(model, image, task_id, scale_id, device=device)

    # Create visualization
    print("Creating visualization...")
    overlay, metrics = create_overlay_visualization(image, ground_truth, prediction, alpha=0.6)

    # Print metrics
    print("\nMetrics:")
    print(f" True Positives: {metrics['tp']:6d} pixels (Blue)")
    print(f" False Positives: {metrics['fp']:6d} pixels (Yellow)")
    print(f" False Negatives: {metrics['fn']:6d} pixels (Magenta)")
    print(f" Precision: {metrics['precision']:.4f}")
    print(f" Recall: {metrics['recall']:.4f}")
    print(f" Dice: {metrics['dice']:.4f}")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(image)
    axes[1].imshow(ground_truth, alpha=0.5, cmap='Greens')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(image)
    axes[2].imshow(prediction * 255, alpha=0.5, cmap='Blues')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Overlay with error visualization
    axes[3].imshow(overlay)
    axes[3].set_title(f'Overlay (Dice={metrics["dice"]:.3f})')
    axes[3].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label=f'True Positive ({metrics["tp"]} px)'),
        Patch(facecolor='yellow', alpha=0.6, label=f'False Positive ({metrics["fp"]} px)'),
        Patch(facecolor='magenta', alpha=0.6, label=f'False Negative ({metrics["fn"]} px)')
    ]
    axes[3].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.suptitle(f'{class_name} Segmentation Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save or show
    if output_path:
        # If output_path is a directory, create a filename
        if os.path.isdir(output_path):
            output_file = os.path.join(output_path, f'{class_name.lower()}_visualization.png')
        else:
            output_file = output_path

        # Create parent directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions for TUFT and CAP')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask', type=str, required=True, help='Path to ground truth mask')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--task_id', type=int, required=True, choices=[0, 1, 2, 3, 4, 5],
                        help='Task ID: 0=DT, 1=PT, 2=CAP, 3=TUFT, 4=VES, 5=PTC')
    parser.add_argument('--scale_id', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Scale ID: 0=5x, 1=10x, 2=20x, 3=40x (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization (default: display only)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Run visualization
    visualize(
        image_path=args.image,
        mask_path=args.mask,
        model_path=args.model,
        task_id=args.task_id,
        scale_id=args.scale_id,
        output_path=args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
