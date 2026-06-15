"""
Dataset Classes for Multi-Scale Kidney Pathology Segmentation

Features:
1. Scale-aware augmentation: Different augmentation strategies per scale
2. Class-balanced sampling: Oversamples minority classes (PTC)
3. Grouped batch sampling: Ensures same-class batches for token-based models
4. Multi-stain support: Handles H&E, PAS, SIL, TRI stains
5. Patch extraction: 512x512 patches from 3000x3000 images
6. Sliding window validation: Full image inference without mask guidance
"""

from .pathology_dataset import (
    PathologyDataset,
    PathologyValDataset,
    PathologyValDatasetCustom,  # Backward compatibility alias
    FullImageValDataset,
    ClassBalancedSampler,
    GroupedBatchSampler,
    get_train_transforms,
    get_val_transforms,
    collate_fn
)

__all__ = [
    'PathologyDataset',
    'PathologyValDataset',
    'PathologyValDatasetCustom',  # Backward compatibility alias
    'FullImageValDataset',
    'ClassBalancedSampler',
    'GroupedBatchSampler',
    'get_train_transforms',
    'get_val_transforms',
    'collate_fn'
]

