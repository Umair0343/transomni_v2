"""
Loss Functions for Multi-Scale Kidney Pathology Segmentation

Addresses key challenges:
1. Class Imbalance: PTC has fewer images but many small objects
2. Object Size Variation: From large TUFT to small PTC
3. Boundary Precision: Important for small structures
4. Multi-Task Learning: Simultaneous segmentation and boundary detection

Loss Components:
- Dice Loss: For handling class imbalance
- Focal Loss: For hard example mining
- Boundary Loss: For precise boundary segmentation
- Class-Balanced Loss: Reweights based on class frequency
"""

from .combined_loss import (
    CombinedLoss,
    DiceLoss,
    TverskyLoss,
    FocalTverskyLoss,
    FocalLoss,
    BoundaryLoss,
    ClassBalancedLoss,
    ConfusionPenaltyLoss,
    OnlineMixup,
    RenalAnatomyLoss,
    get_class_weights
)
from .hats_loss import HATsLoss, get_hats_loss

__all__ = [
    'CombinedLoss',
    'DiceLoss',
    'TverskyLoss',
    'FocalTverskyLoss',
    'FocalLoss',
    'BoundaryLoss',
    'ClassBalancedLoss',
    'ConfusionPenaltyLoss',
    'OnlineMixup',
    'RenalAnatomyLoss',
    'get_class_weights',
    'HATsLoss',
    'get_hats_loss'
]
