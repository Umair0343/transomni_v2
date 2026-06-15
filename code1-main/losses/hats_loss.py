"""
Simplified Loss for HATs Model

The original HATs uses a straightforward loss approach:
- Cross-Entropy for classification
- Dice for class imbalance
- Class-balanced weighting (not per-loss multiplication)

This simplified version avoids:
- Redundant losses (Dice + Tversky doing the same thing)
- Over-complicated weighting schemes
- Unused boundary loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class DiceLoss(nn.Module):
    """Simple Dice Loss for binary segmentation"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits [B, 2, H, W]
            target: Ground truth [B, H, W]
        """
        # Get foreground probability
        pred_fg = torch.sigmoid(pred[:, 1])  # [B, H, W]
        target_fg = target.float()

        # Flatten
        pred_flat = pred_fg.view(-1)
        target_flat = target_fg.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CEDiceLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice Loss

    This is a standard, stable loss for segmentation:
    - CE provides pixel-wise gradients
    - Dice handles class imbalance globally
    """

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits [B, 2, H, W]
            target: Ground truth [B, H, W] (0 or 1)
        """
        # Cross-entropy
        ce = F.cross_entropy(pred, target.long())

        # Dice
        dice = self.dice_loss(pred, target)

        return self.ce_weight * ce + self.dice_weight * dice


class HATsLoss(nn.Module):
    """
    Simplified Loss Function for HATs Model

    Components:
    1. Cross-Entropy + Dice (standard, stable)
    2. Class-balanced weighting (based on foreground pixel %)
    3. Optional scale weighting (gentle, not aggressive)

    This is simpler and more stable than the complex CombinedLoss.
    """

    # Class weights based on foreground pixel percentage (inverse)
    # VES (0.89%) gets highest weight, PT (34.79%) gets lowest
    CLASS_WEIGHTS = torch.tensor([
        1.0,   # DT (28.92% FG)
        0.8,   # PT (34.79% FG) - most foreground, needs less weight
        1.5,   # CAP (11.23% FG)
        1.8,   # TUFT (9.18% FG)
        3.0,   # VES (0.89% FG) - very sparse, needs boost
        2.0,   # PTC (4.62% FG)
    ])

    # Gentle scale weights
    SCALE_WEIGHTS = torch.tensor([1.0, 1.0, 1.1, 1.2])  # 5x, 10x, 20x, 40x

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        use_class_balance: bool = True,
        use_scale_balance: bool = True
    ):
        super().__init__()

        self.use_class_balance = use_class_balance
        self.use_scale_balance = use_scale_balance

        self.ce_dice_loss = CEDiceLoss(ce_weight, dice_weight)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task_id: torch.Tensor,
        scale_id: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None  # Unused, for compatibility
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.

        Args:
            outputs: Model outputs with 'seg_logits' [B, 2, H, W]
            targets: Ground truth [B, H, W]
            task_id: Task identifier
            scale_id: Scale identifier
            edge_weight: Unused (for API compatibility)

        Returns:
            loss: Scalar loss
            loss_dict: Dictionary of loss components
        """
        seg_logits = outputs['seg_logits']

        # Base loss (CE + Dice)
        base_loss = self.ce_dice_loss(seg_logits, targets)

        # Get weights
        task_idx = task_id[0].long().item() if task_id.numel() > 0 else 0
        scale_idx = scale_id[0].long().item() if scale_id.numel() > 0 else 0
        task_idx = min(task_idx, len(self.CLASS_WEIGHTS) - 1)
        scale_idx = min(scale_idx, len(self.SCALE_WEIGHTS) - 1)

        class_weight = self.CLASS_WEIGHTS[task_idx].to(seg_logits.device) if self.use_class_balance else 1.0
        scale_weight = self.SCALE_WEIGHTS[scale_idx].to(seg_logits.device) if self.use_scale_balance else 1.0

        # Apply weights (multiplicative, but gentler than before)
        total_loss = base_loss * class_weight * scale_weight

        # Compute dice for logging
        with torch.no_grad():
            pred_fg = torch.sigmoid(seg_logits[:, 1]) > 0.5
            target_fg = targets.bool()
            intersection = (pred_fg & target_fg).sum().float()
            union = pred_fg.sum().float() + target_fg.sum().float()
            dice_score = (2.0 * intersection + 1.0) / (union + 1.0)

        loss_dict = {
            'total': total_loss.item(),
            'base_loss': base_loss.item(),
            'class_weight': float(class_weight) if isinstance(class_weight, torch.Tensor) else class_weight,
            'scale_weight': float(scale_weight) if isinstance(scale_weight, torch.Tensor) else scale_weight,
            'dice_score': dice_score.item()
        }

        return total_loss, loss_dict


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss - differentiable version without thresholding.

    Better for training as gradients flow through all predictions.
    """

    def __init__(self, smooth: float = 1.0, square: bool = False):
        super().__init__()
        self.smooth = smooth
        self.square = square

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = torch.softmax(pred, dim=1)[:, 1]  # Foreground probability
        target_f = target.float()

        if self.square:
            # Squared Dice (stronger gradient for small objects)
            intersection = (pred_soft * target_f).sum()
            union = (pred_soft ** 2).sum() + (target_f ** 2).sum()
        else:
            intersection = (pred_soft * target_f).sum()
            union = pred_soft.sum() + target_f.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


def get_hats_loss(use_class_balance: bool = True, exclude_classes: list = None) -> HATsLoss:
    """Factory function for HATs loss

    Args:
        use_class_balance: Whether to use class-balanced weighting
        exclude_classes: List of class IDs to exclude (unused in this simplified loss,
                        but accepted for API consistency with CombinedLoss)
    """
    return HATsLoss(
        ce_weight=0.5,
        dice_weight=0.5,
        use_class_balance=use_class_balance,
        use_scale_balance=True
    )


if __name__ == '__main__':
    # Test
    print("Testing HATsLoss...")

    loss_fn = get_hats_loss(use_class_balance=True)

    B, H, W = 2, 512, 512
    pred = torch.randn(B, 2, H, W)
    target = torch.randint(0, 2, (B, H, W))

    outputs = {'seg_logits': pred}

    # Test all classes
    class_names = ['DT', 'PT', 'CAP', 'TUFT', 'VES', 'PTC']
    print("\nLoss per class:")
    for task_idx in range(6):
        task_id = torch.tensor([task_idx])
        scale_id = torch.tensor([1])
        loss, loss_dict = loss_fn(outputs, target, task_id, scale_id)
        print(f"  {class_names[task_idx]}: loss={loss.item():.4f}, "
              f"class_weight={loss_dict['class_weight']:.2f}")

    print("\nTest passed!")
