"""
Combined Loss Functions for Multi-Scale Segmentation

Implements multiple loss components to address:
1. Class Imbalance: Weighted losses based on class frequency
2. Hard Examples: Focal loss for difficult pixels
3. Boundary Precision: Explicit boundary supervision
4. Multi-Scale Objects: Scale-aware loss weighting

Reference:
- Focal Loss (ICCV 2017): For hard example mining
- Dice Loss: For class imbalance
- Boundary Loss (MICCAI 2019): For precise boundaries
- Class-Balanced Loss (CVPR 2019): For long-tailed distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


def get_class_weights(
    num_samples_per_class: Optional[Dict[int, int]] = None,
    use_pixel_weights: bool = True,
    beta: float = 0.999,
    exclude_classes: Optional[list] = None
) -> torch.Tensor:
    """
    Compute class-balanced weights based on effective number of samples
    AND pixel-level foreground distribution.

    Based on "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    Enhanced with pixel-level analysis for kidney pathology.

    Args:
        num_samples_per_class: Dictionary mapping class_id to number of samples
        use_pixel_weights: Whether to incorporate pixel-level imbalance
        beta: Hyperparameter (default 0.999)
        exclude_classes: List of class IDs to exclude (e.g., [4, 5] for VES, PTC)

    Returns:
        Tensor of class weights
    """
    if num_samples_per_class is None:
        # Default for kidney pathology dataset
        # Based on training data analysis:
        # DT: 285, PT: 285, CAP: 369, TUFT: 388, VES: 182, PTC: 96
        num_samples_per_class = {
            0: 285,   # DT
            1: 285,   # PT
            2: 369,   # CAP
            3: 388,   # TUFT
            4: 182,   # VES
            5: 96     # PTC (fewest samples)
        }

    # Filter out excluded classes
    if exclude_classes:
        num_samples_per_class = {k: v for k, v in num_samples_per_class.items() if k not in exclude_classes}

    num_classes = len(num_samples_per_class)
    active_classes = sorted(num_samples_per_class.keys())
    samples = np.array([num_samples_per_class[i] for i in active_classes])

    # Effective number of samples
    effective_num = 1.0 - np.power(beta, samples)
    sample_weights = (1.0 - beta) / effective_num

    if use_pixel_weights:
        # Pixel-level foreground percentage from mask analysis:
        # PT: 34.79%, DT: 28.92%, CAP: 11.23%, TUFT: 9.18%, PTC: 4.62%, ART: 0.89%
        all_pixel_fg_percent = {
            0: 28.92,  # DT
            1: 34.79,  # PT
            2: 11.23,  # CAP
            3: 9.18,   # TUFT
            4: 0.89,   # ART/VES - VERY sparse!
            5: 4.62,   # PTC
        }
        pixel_fg_percent = np.array([all_pixel_fg_percent[i] for i in active_classes])

        # Inverse of pixel percentage (higher weight for sparse classes)
        pixel_weights = 1.0 / (pixel_fg_percent + 1e-6)
        pixel_weights = pixel_weights / pixel_weights.sum() * num_classes

        # Combine sample-level and pixel-level weights
        # Use geometric mean to balance both factors
        weights = np.sqrt(sample_weights * pixel_weights)

        # Additional boost for hardest classes if they are active
        # DT: scattered tiny blobs, morphologically harder than pixel coverage suggests
        if 0 in active_classes:  # DT - weakest class, scattered morphology
            dt_idx = active_classes.index(0)
            weights[dt_idx] *= 1.3
        if 4 in active_classes:  # VES - most sparse class
            ves_idx = active_classes.index(4)
            weights[ves_idx] *= 1.5
        if 3 in active_classes:  # TUFT - large but difficult
            tuft_idx = active_classes.index(3)
            weights[tuft_idx] *= 1.2
    else:
        weights = sample_weights

    # Normalize
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def get_pixel_aware_weights() -> torch.Tensor:
    """
    Get weights based purely on pixel-level foreground distribution.

    Derived from actual mask analysis:
    - PT: 34.79% FG (dense tubules)
    - DT: 28.92% FG (dense tubules)
    - CAP: 11.23% FG (large glomerular)
    - TUFT: 9.18% FG (large glomerular)
    - PTC: 4.62% FG (sparse tiny objects)
    - ART: 0.89% FG (very sparse vessels)

    Returns:
        Tensor of pixel-aware weights [6]
    """
    # Foreground percentages from mask analysis
    fg_percent = torch.tensor([
        28.92,  # DT
        34.79,  # PT
        11.23,  # CAP
        9.18,   # TUFT
        0.89,   # ART - needs highest weight!
        4.62,   # PTC
    ])

    # Inverse weighting (sparse classes get higher weight)
    weights = 1.0 / fg_percent

    # Normalize to mean=1
    weights = weights / weights.mean()

    # Cap extreme weights to avoid instability
    weights = torch.clamp(weights, min=0.1, max=20.0)

    return weights


class DiceLoss(nn.Module):
    """
    Dice Loss for binary/multi-class segmentation.

    Handles class imbalance by directly optimizing the Dice coefficient.
    """
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth [B, H, W] or [B, C, H, W]
            weight: Optional pixel-wise weights [B, H, W]
        """
        # Apply sigmoid for binary classification
        pred = torch.sigmoid(pred)

        # Handle different target formats
        if target.dim() == 3:
            # Convert to one-hot if needed [B, H, W] -> [B, C, H, W]
            num_classes = pred.shape[1]
            target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        else:
            target_onehot = target

        # Apply weight if provided
        if weight is not None:
            weight = weight.unsqueeze(1)  # [B, 1, H, W]
            pred = pred * weight
            target_onehot = target_onehot * weight

        # Compute dice per class
        dims = (0, 2, 3)  # Batch and spatial dims
        intersection = (pred * target_onehot).sum(dim=dims)
        union = pred.sum(dim=dims) + target_onehot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss that handles class imbalance better.

    Controls trade-off between false positives and false negatives:
    - alpha > beta: penalize false negatives more (good for sparse classes like VES)
    - alpha < beta: penalize false positives more
    - alpha = beta = 0.5: equivalent to Dice loss

    Reference: "Tversky Loss Function for Image Segmentation" (2017)
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight (higher = penalize missing foreground more)
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        if target.dim() == 3:
            num_classes = pred.shape[1]
            target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        else:
            target_onehot = target

        if weight is not None:
            weight = weight.unsqueeze(1)
            pred = pred * weight
            target_onehot = target_onehot * weight

        dims = (0, 2, 3)
        tp = (pred * target_onehot).sum(dim=dims)
        fp = (pred * (1 - target_onehot)).sum(dim=dims)
        fn = ((1 - pred) * target_onehot).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return (1 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - combines Tversky with focal mechanism for very hard examples.

    gamma > 1: focus more on hard examples (misclassified pixels)

    Reference: "A Novel Focal Tversky Loss Function for Imbalanced Data" (2019)
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # Focal parameter
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        if target.dim() == 3:
            num_classes = pred.shape[1]
            target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        else:
            target_onehot = target

        if weight is not None:
            weight = weight.unsqueeze(1)
            pred = pred * weight
            target_onehot = target_onehot * weight

        dims = (0, 2, 3)
        tp = (pred * target_onehot).sum(dim=dims)
        fp = (pred * (1 - target_onehot)).sum(dim=dims)
        fn = ((1 - pred) * target_onehot).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        # Clamp to prevent numerical instability
        tversky = torch.clamp(tversky, min=1e-7, max=1.0 - 1e-7)
        focal_tversky = (1 - tversky).pow(self.gamma)
        # Clamp final loss to prevent NaN
        focal_tversky = torch.clamp(focal_tversky, min=0.0, max=10.0)

        return focal_tversky.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples.

    Reduces the loss contribution from easy examples and focuses
    learning on hard negatives.

    Reference: "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth [B, H, W]
            weight: Optional pixel-wise weights
        """
        # Convert to binary classification format if needed
        if pred.shape[1] == 2:
            # Use class 1 (foreground) probability
            pred_prob = torch.sigmoid(pred[:, 1:2, :, :])  # [B, 1, H, W]
            target = target.unsqueeze(1).float()  # [B, 1, H, W]
        else:
            pred_prob = torch.sigmoid(pred)
            target = target.float()

        # Focal loss computation
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            pred[:, 1:2, :, :] if pred.shape[1] == 2 else pred,
            target if target.dim() == 4 else target.unsqueeze(1).float(),
            reduction='none'
        )

        loss = focal_weight * bce

        if weight is not None:
            loss = loss * weight.unsqueeze(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for precise edge segmentation.

    Particularly important for small structures like PTC where
    boundary accuracy is crucial.

    Reference: "Boundary Loss for Highly Unbalanced Segmentation" (MICCAI 2019)
    """
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def compute_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute boundary mask using morphological operations"""
        # Use max pooling to dilate and negative max pooling to erode
        kernel_size = 3
        padding = kernel_size // 2

        mask_float = mask.float().unsqueeze(1)  # [B, 1, H, W]

        # Dilation
        dilated = F.max_pool2d(mask_float, kernel_size, stride=1, padding=padding)

        # Erosion (using negative)
        eroded = -F.max_pool2d(-mask_float, kernel_size, stride=1, padding=padding)

        # Boundary = dilated - eroded
        boundary = (dilated - eroded).squeeze(1)

        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth [B, H, W]
        """
        # Get predicted class
        pred_softmax = F.softmax(pred, dim=1)
        pred_class = pred_softmax[:, 1, :, :]  # Foreground probability

        # Compute boundaries
        pred_boundary = self.compute_boundary((pred_class > 0.5).float())
        gt_boundary = self.compute_boundary(target.float())

        # Extended boundary for loss computation
        gt_boundary_ext = F.max_pool2d(
            gt_boundary.unsqueeze(1),
            kernel_size=int(self.theta0),
            stride=1,
            padding=int(self.theta0) // 2
        ).squeeze(1)

        # Boundary-aware loss
        # Penalize predictions that are on the boundary but incorrect
        boundary_diff = torch.abs(pred_boundary - gt_boundary)
        boundary_loss = (boundary_diff * gt_boundary_ext).mean()

        return boundary_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-balanced wrapper that applies different weights per task class.

    Addresses the imbalance where PTC has fewer images but dense annotations.
    """
    def __init__(self, base_loss: nn.Module, num_classes: int = 6,
                 beta: float = 0.999):
        super().__init__()
        self.base_loss = base_loss
        self.class_weights = get_class_weights(beta=beta)
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                task_id: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits
            target: Ground truth
            task_id: Task/class identifier
            weight: Optional pixel-wise weights
        """
        # Get class weight
        task_idx = task_id[0].long().item() if task_id.numel() > 0 else 0
        task_idx = min(task_idx, self.num_classes - 1)

        class_weight = self.class_weights[task_idx].to(pred.device)

        # Compute base loss
        loss = self.base_loss(pred, target, weight)

        # Apply class weight
        return loss * class_weight


class ConfusionPenaltyLoss(nn.Module):
    """
    Confusion Penalty Loss for visually similar classes (DT vs PT).

    More targeted than generic exclusive anatomy loss — specifically penalizes
    when one class's foreground overlaps with a confusable class's prediction.

    Uses asymmetric Dice overlap penalty with higher weight than anatomy exclusive.
    """
    DT = 0
    PT = 1

    def __init__(self, penalty_weight: float = 0.5, smooth: float = 1.0):
        """
        Args:
            penalty_weight: Weight for the confusion penalty
            smooth: Smoothing factor
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        self.smooth = smooth

        # Define confusion pairs: (class_a, class_b)
        self.confusion_pairs = [
            (self.DT, self.PT),  # DT and PT look similar
        ]

    def get_confused_class(self, current_class: int) -> Optional[int]:
        """Get the confusable class for a given class, or None."""
        for a, b in self.confusion_pairs:
            if current_class == a:
                return b
            if current_class == b:
                return a
        return None

    def forward(
        self,
        current_label: torch.Tensor,
        confused_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confusion penalty.

        Args:
            current_label: Ground truth for current class [B, H, W] (binary 0/1)
            confused_pred: Prediction probability for confused class [B, H, W] (0-1)

        Returns:
            Scalar penalty loss
        """
        current_label = current_label.float()
        confused_pred = confused_pred.float()

        # Penalize overlap: confused_pred should be LOW where current_label is HIGH
        # This is stronger than anatomy exclusive — uses squared overlap
        intersection = (current_label * confused_pred).sum()
        total = current_label.sum() + confused_pred.sum()

        # Squared overlap ratio for stronger penalty
        overlap = (2 * intersection + self.smooth) / (total + self.smooth)
        penalty = overlap ** 2  # Squared for stronger gradient on high overlap

        return penalty * self.penalty_weight


class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-scale segmentation.

    Combines:
    1. Dice Loss: For class imbalance
    2. Focal Loss: For hard examples
    3. Boundary Loss: For precise boundaries
    4. Auxiliary Loss: For deep supervision

    With class-balanced weighting based on task_id.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        boundary_weight: float = 0.5,
        tversky_weight: float = 0.5,  # Reduced from 1.0 for stability
        aux_weight: float = 0.4,
        num_task_classes: int = 6,
        use_class_balance: bool = True,
        exclude_classes: Optional[list] = None
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.tversky_weight = tversky_weight
        self.aux_weight = aux_weight
        self.use_class_balance = use_class_balance
        self.exclude_classes = exclude_classes

        # Loss components
        self.dice_loss = DiceLoss(smooth=1.0)
        # Focal Tversky: alpha=0.3, beta=0.7 penalizes false negatives more (missing foreground)
        # This helps with sparse classes like VES
        self.focal_tversky_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)  # Reduced gamma for stability
        # Conservative focal loss settings
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)  # Back to original stable values
        self.boundary_loss = BoundaryLoss()

        # Class weights for task-aware weighting
        if use_class_balance:
            self.class_weights = get_class_weights(exclude_classes=exclude_classes)
        else:
            self.class_weights = torch.ones(num_task_classes)

        # Learnable scale-specific weights (initialized to hand-tuned values)
        # Scale 0 (5x): large objects, Scale 3 (40x): small objects
        # Uses softplus to keep weights positive; raw values chosen so softplus(raw) ≈ target
        import math
        init_targets = [1.0, 1.2, 1.5, 2.0]
        raw_init = [math.log(math.exp(t) - 1) for t in init_targets]  # inverse softplus
        self.scale_weights_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task_id: torch.Tensor,
        scale_id: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs dictionary containing:
                - 'seg_logits': [B, 2, H, W]
                - 'boundary_logits': [B, 1, H, W]
                - 'aux_logits': [B, 2, H, W] (optional)
            targets: Ground truth segmentation [B, H, W]
            task_id: Task identifier
            scale_id: Scale identifier
            edge_weight: Optional edge weight map

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        seg_logits = outputs['seg_logits']
        boundary_logits = outputs.get('boundary_logits')
        aux_logits = outputs.get('aux_logits')

        # Get class and scale weights
        task_idx = task_id[0].long().item() if task_id.numel() > 0 else 0
        scale_idx = scale_id[0].long().item() if scale_id.numel() > 0 else 0
        task_idx = min(task_idx, len(self.class_weights) - 1)
        scale_idx = min(scale_idx, len(self.scale_weights_raw) - 1)

        class_weight = self.class_weights[task_idx].to(seg_logits.device)
        # Learnable scale weights via softplus (always positive)
        scale_weights = F.softplus(self.scale_weights_raw)
        scale_weight = scale_weights[scale_idx]

        # Combined task+scale weight
        combined_weight = class_weight * scale_weight

        loss_dict = {}
        total_loss = 0.0

        # Dice loss
        dice = self.dice_loss(seg_logits, targets, edge_weight)
        loss_dict['dice'] = dice.item()
        total_loss = total_loss + self.dice_weight * dice * combined_weight

        # Focal Tversky loss (better for imbalanced segmentation)
        focal_tversky = self.focal_tversky_loss(seg_logits, targets, edge_weight)
        loss_dict['focal_tversky'] = focal_tversky.item()
        total_loss = total_loss + self.tversky_weight * focal_tversky * combined_weight

        # Focal loss
        focal = self.focal_loss(seg_logits, targets, edge_weight)
        loss_dict['focal'] = focal.item()
        total_loss = total_loss + self.focal_weight * focal * combined_weight

        # Boundary loss
        if boundary_logits is not None:
            boundary = self.boundary_loss(seg_logits, targets)
            loss_dict['boundary'] = boundary.item()
            total_loss = total_loss + self.boundary_weight * boundary * combined_weight

        # Auxiliary loss (deep supervision)
        if aux_logits is not None:
            aux_dice = self.dice_loss(aux_logits, targets)
            aux_focal = self.focal_loss(aux_logits, targets)
            aux_loss = aux_dice + 0.5 * aux_focal
            loss_dict['aux'] = aux_loss.item()
            total_loss = total_loss + self.aux_weight * aux_loss * combined_weight

        loss_dict['total'] = total_loss.item()
        loss_dict['class_weight'] = class_weight.item()
        loss_dict['scale_weight'] = scale_weight.item()

        return total_loss, loss_dict


class OnlineMixup(nn.Module):
    """
    Online MixUp augmentation for regularization.

    Helps prevent overfitting by mixing training samples.
    """
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Mix batch samples"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index].float()

        return mixed_x, mixed_y, lam


class RenalAnatomyLoss(nn.Module):
    """
    Renal Anatomy Loss based on Universal Proposition Learning (PrPSeg).

    Enforces anatomical constraints between kidney structures using a
    Universal Proposition Matrix M that defines relationships:
    - M = 1 (Subset): Class i is anatomically inside Class j
    - M = -1 (Superset): Class i anatomically contains Class j
    - M = 2 (Exclusive): Classes never overlap

    Classes: 0=DT, 1=PT, 2=CAP, 3=TUFT, 4=VES, 5=PTC

    Key anatomical relationships:
    - TUFT is inside CAP (glomerular tuft within Bowman's capsule)
    - DT and PT are exclusive (different tubule segments)
    - Tubules (DT, PT) are exclusive with Glomeruli (CAP, TUFT)

    Reference: "Universal Proposition Learning for Pathology Segmentation"
    """
    # Class indices
    DT = 0
    PT = 1
    CAP = 2
    TUFT = 3
    VES = 4
    PTC = 5

    # Class names for logging
    CLASS_NAMES = ['DT', 'PT', 'CAP', 'TUFT', 'VES', 'PTC']

    def __init__(self, weight: float = 0.3, smooth: float = 1.0):
        """
        Args:
            weight: Weight for this loss in the combined loss
            smooth: Smoothing factor to prevent division by zero
        """
        super().__init__()
        self.weight = weight
        self.smooth = smooth

        # Build and register the relationship matrix
        self.register_buffer('M', self._build_relationship_matrix())

        # Define which classes have meaningful relationships
        # Only compute loss when current class has related classes
        self.related_classes = self._build_related_classes()

    def _build_relationship_matrix(self) -> torch.Tensor:
        """
        Build the Universal Proposition Matrix M.

        M[i, j] defines the relationship when:
        - Label (ground truth) is for class i
        - Prediction is for class j

        Values:
        - 0: No constraint
        - 1: Subset (class i is inside class j)
        - -1: Superset (class i contains class j)
        - 2: Exclusive (classes never overlap)
        """
        M = torch.zeros(6, 6, dtype=torch.float32)

        # ===== TUFT (3) vs CAP (2) - Glomerular relationship =====
        # TUFT is anatomically inside CAP (Bowman's capsule)
        M[self.TUFT, self.CAP] = 1    # When training TUFT, it should be inside CAP prediction
        M[self.CAP, self.TUFT] = -1   # When training CAP, TUFT prediction should be inside it

        # ===== Tubules (DT, PT) are mutually exclusive =====
        M[self.DT, self.PT] = 2
        M[self.PT, self.DT] = 2

        # ===== Tubules vs Glomeruli are exclusive =====
        for tubule in [self.DT, self.PT]:
            for glomerulus in [self.CAP, self.TUFT]:
                M[tubule, glomerulus] = 2
                M[glomerulus, tubule] = 2

        # ===== VES (vessels) exclusive with everything except itself =====
        for i in range(6):
            if i != self.VES:
                M[self.VES, i] = 2
                M[i, self.VES] = 2

        # ===== PTC (peritubular capillaries) exclusive with everything except itself =====
        for i in range(6):
            if i != self.PTC:
                M[self.PTC, i] = 2
                M[i, self.PTC] = 2

        return M

    def _build_related_classes(self) -> Dict[int, list]:
        """
        Build a dictionary of related classes for each class.

        Only includes classes with non-zero relationships.
        Returns dict: class_id -> list of (related_class_id, relationship_type)
        """
        related = {}
        M = self._build_relationship_matrix()

        for i in range(6):
            related[i] = []
            for j in range(6):
                if i != j and M[i, j] != 0:
                    related[i].append((j, int(M[i, j].item())))

        return related

    def get_related_classes(self, class_id: int) -> list:
        """Get list of (related_class_id, relationship_type) for a given class."""
        return self.related_classes.get(class_id, [])

    def get_primary_related_class(self, class_id: int) -> Optional[Tuple[int, int]]:
        """
        Get the most important related class for a given class.

        Priority: Subset/Superset (±1) over Exclusive (2)
        Returns: (related_class_id, relationship_type) or None
        """
        related = self.related_classes.get(class_id, [])
        if not related:
            return None

        # Prioritize subset/superset relationships (more informative)
        for rel_class, rel_type in related:
            if rel_type in [1, -1]:
                return (rel_class, rel_type)

        # Fall back to first exclusive relationship
        return related[0] if related else None

    def forward(
        self,
        label_i: torch.Tensor,
        pred_j: torch.Tensor,
        class_i: int,
        class_j: int
    ) -> torch.Tensor:
        """
        Compute anatomy loss for a pair of classes.

        Args:
            label_i: Ground truth for class i [B, H, W] (binary 0/1)
            pred_j: Prediction probability for class j [B, H, W] (0-1)
            class_i: Index of class i (ground truth class)
            class_j: Index of class j (prediction class)

        Returns:
            Scalar loss value
        """
        # Get relationship from matrix
        relation = int(self.M[class_i, class_j].item())

        if relation == 0:
            return torch.tensor(0.0, device=label_i.device, dtype=label_i.dtype)

        # Ensure proper tensor format
        label_i = label_i.float()
        pred_j = pred_j.float()

        if relation == 1:
            # SUBSET: label_i should be INSIDE pred_j
            # Example: TUFT (label) should be inside CAP (prediction)
            # Loss = pixels of label that fall outside prediction
            return self._subset_loss(label_i, pred_j)

        elif relation == -1:
            # SUPERSET: pred_j should be INSIDE label_i
            # Example: TUFT (prediction) should be inside CAP (label)
            # Loss = pixels of prediction that fall outside label
            return self._superset_loss(label_i, pred_j)

        elif relation == 2:
            # EXCLUSIVE: label_i and pred_j should NOT overlap
            # Example: DT (label) and PT (prediction) should not overlap
            # Loss = intersection / union (want to minimize overlap)
            return self._exclusive_loss(label_i, pred_j)

        return torch.tensor(0.0, device=label_i.device, dtype=label_i.dtype)

    def _subset_loss(self, label: torch.Tensor, container_pred: torch.Tensor) -> torch.Tensor:
        """
        Subset loss: label should be inside container_pred.

        Logic: Minimize |label - (label ∩ container_pred)|
        This penalizes when label pixels fall outside the container prediction.

        Args:
            label: Ground truth mask (the smaller structure, e.g., TUFT)
            container_pred: Prediction for containing structure (e.g., CAP)
        """
        # Intersection (soft)
        intersection = label * container_pred

        # Pixels of label outside container
        outside = label - intersection

        # Normalize by label size (what fraction of label is outside?)
        label_sum = label.sum() + self.smooth
        loss = outside.abs().sum() / label_sum

        return loss

    def _superset_loss(self, container_label: torch.Tensor, contained_pred: torch.Tensor) -> torch.Tensor:
        """
        Superset loss: contained_pred should be inside container_label.

        Logic: Minimize |contained_pred - (contained_pred ∩ container_label)|
        This penalizes when prediction pixels fall outside the container ground truth.

        Args:
            container_label: Ground truth for containing structure (e.g., CAP)
            contained_pred: Prediction for contained structure (e.g., TUFT)
        """
        # Intersection (soft)
        intersection = container_label * contained_pred

        # Pixels of prediction outside container
        outside = contained_pred - intersection

        # Normalize by prediction size
        pred_sum = contained_pred.sum() + self.smooth
        loss = outside.abs().sum() / pred_sum

        return loss

    def _exclusive_loss(self, label_a: torch.Tensor, pred_b: torch.Tensor) -> torch.Tensor:
        """
        Exclusive loss: label_a and pred_b should NOT overlap.

        Logic: Minimize Dice intersection between them.

        Args:
            label_a: Ground truth for one class (e.g., DT)
            pred_b: Prediction for another class (e.g., PT)
        """
        # Soft intersection
        intersection = (label_a * pred_b).sum()
        total = label_a.sum() + pred_b.sum()

        # We want intersection to be 0
        # Return overlap ratio (higher = worse)
        overlap_ratio = (2 * intersection + self.smooth) / (total + self.smooth)

        # For exclusive classes, we want overlap_ratio to be 0
        # So loss = overlap_ratio (penalize any overlap)
        return overlap_ratio

    def compute_batch_loss(
        self,
        current_label: torch.Tensor,
        related_predictions: Dict[int, torch.Tensor],
        current_class: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute anatomy loss for a batch, given predictions for related classes.

        Args:
            current_label: Ground truth for current class [B, H, W]
            related_predictions: Dict mapping class_id -> prediction [B, H, W]
            current_class: Current class index

        Returns:
            total_loss: Combined anatomy loss
            loss_dict: Individual losses per relationship
        """
        total_loss = torch.tensor(0.0, device=current_label.device)
        loss_dict = {}

        related = self.get_related_classes(current_class)

        for rel_class, rel_type in related:
            if rel_class not in related_predictions:
                continue

            pred_j = related_predictions[rel_class]
            loss = self.forward(current_label, pred_j, current_class, rel_class)

            # Log with meaningful names
            rel_name = f"{self.CLASS_NAMES[current_class]}_vs_{self.CLASS_NAMES[rel_class]}"
            rel_type_name = {1: "subset", -1: "superset", 2: "exclusive"}[rel_type]
            loss_dict[f"anatomy_{rel_name}_{rel_type_name}"] = loss.item()

            total_loss = total_loss + loss

        return total_loss, loss_dict


if __name__ == '__main__':
    # Test losses
    print("Testing loss functions...")

    B, C, H, W = 2, 2, 512, 512

    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, H, W))
    task_id = torch.tensor([5])  # PTC (hardest class)
    scale_id = torch.tensor([3])  # 40x (smallest objects)

    outputs = {
        'seg_logits': pred,
        'boundary_logits': torch.randn(B, 1, H, W),
        'aux_logits': torch.randn(B, C, H, W)
    }

    loss_fn = CombinedLoss(
        dice_weight=1.0,
        focal_weight=0.5,
        boundary_weight=0.5,
        aux_weight=0.4,
        num_task_classes=6,
        use_class_balance=True
    )

    total_loss, loss_dict = loss_fn(outputs, target, task_id, scale_id)

    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\nClass weights:")
    weights = get_class_weights()
    class_names = ['DT', 'PT', 'CAP', 'TUFT', 'VES', 'PTC']
    for i, (name, w) in enumerate(zip(class_names, weights)):
        print(f"  {name}: {w:.4f}")

    print("\nCombinedLoss test passed!")

    # Test RenalAnatomyLoss
    print("\n" + "="*60)
    print("Testing RenalAnatomyLoss...")
    print("="*60)

    anatomy_loss = RenalAnatomyLoss(weight=0.3)

    # Print relationship matrix
    print("\nRelationship Matrix M (1=subset, -1=superset, 2=exclusive):")
    print("       ", "  ".join(class_names))
    for i, name_i in enumerate(class_names):
        row = [f"{int(anatomy_loss.M[i, j].item()):2d}" for j in range(6)]
        print(f"  {name_i}: {' '.join(row)}")

    # Test case 1: TUFT inside CAP (subset relationship)
    print("\nTest 1: TUFT (label) vs CAP (prediction) - SUBSET")
    tuft_label = torch.zeros(1, 64, 64)
    tuft_label[:, 20:40, 20:40] = 1.0  # Small square

    # Case 1a: CAP prediction fully contains TUFT - should have low loss
    cap_pred_good = torch.zeros(1, 64, 64)
    cap_pred_good[:, 15:45, 15:45] = 1.0  # Larger square containing TUFT
    loss_good = anatomy_loss.forward(tuft_label, cap_pred_good, 3, 2)
    print(f"  CAP contains TUFT (good): loss = {loss_good.item():.4f}")

    # Case 1b: CAP prediction doesn't contain TUFT - should have high loss
    cap_pred_bad = torch.zeros(1, 64, 64)
    cap_pred_bad[:, 0:20, 0:20] = 1.0  # Square not overlapping TUFT
    loss_bad = anatomy_loss.forward(tuft_label, cap_pred_bad, 3, 2)
    print(f"  CAP doesn't contain TUFT (bad): loss = {loss_bad.item():.4f}")

    # Test case 2: DT vs PT (exclusive relationship)
    print("\nTest 2: DT (label) vs PT (prediction) - EXCLUSIVE")
    dt_label = torch.zeros(1, 64, 64)
    dt_label[:, 10:30, 10:30] = 1.0

    # Case 2a: PT prediction doesn't overlap DT - should have low loss
    pt_pred_good = torch.zeros(1, 64, 64)
    pt_pred_good[:, 35:55, 35:55] = 1.0  # No overlap
    loss_good = anatomy_loss.forward(dt_label, pt_pred_good, 0, 1)
    print(f"  PT doesn't overlap DT (good): loss = {loss_good.item():.4f}")

    # Case 2b: PT prediction overlaps DT - should have high loss
    pt_pred_bad = torch.zeros(1, 64, 64)
    pt_pred_bad[:, 15:35, 15:35] = 1.0  # Overlaps with DT
    loss_bad = anatomy_loss.forward(dt_label, pt_pred_bad, 0, 1)
    print(f"  PT overlaps DT (bad): loss = {loss_bad.item():.4f}")

    # Test batch loss computation
    print("\nTest 3: Batch loss computation for TUFT")
    related_preds = {
        2: cap_pred_good,  # CAP
        0: torch.zeros(1, 64, 64),  # DT (no prediction)
        1: torch.zeros(1, 64, 64),  # PT (no prediction)
    }
    total_loss, loss_dict = anatomy_loss.compute_batch_loss(tuft_label, related_preds, 3)
    print(f"  Total anatomy loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.4f}")

    print("\nRenalAnatomyLoss test passed!")
