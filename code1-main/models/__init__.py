"""
Hybrid CNN-Transformer Models for Multi-Scale Kidney Pathology Segmentation

Architecture inspired by:
- TransUNet (CVPR 2021): Hybrid CNN-Transformer for medical image segmentation
- SegFormer (NeurIPS 2021): Efficient hierarchical transformer
- Deformable DETR (ICLR 2021): Deformable attention for multi-scale features
- Boundary-Aware Networks (MICCAI): For precise boundary segmentation

Key innovations:
1. Scale-Aware Deformable Attention: Adapts receptive field based on scale_id
2. Class-Conditioned Feature Modulation: Dynamic feature adjustment per class
3. Multi-Scale Feature Pyramid with Boundary Enhancement
4. Anti-overfitting techniques for class imbalance
"""

from .hybrid_scale_transformer import HybridScaleTransformer, build_hybrid_model
from .cnn_encoder import MultiScaleCNNEncoder
from .pretrained_encoder import PretrainedResNetEncoder
from .hats_encoder import HATsModel, build_hats_model
from .transformer_blocks import ScaleAwareTransformerEncoder, DeformableAttention
from .decoder import MultiScaleDecoder, BoundaryAwareHead

__all__ = [
    'HybridScaleTransformer',
    'build_hybrid_model',
    'HATsModel',
    'build_hats_model',
    'MultiScaleCNNEncoder',
    'PretrainedResNetEncoder',
    'ScaleAwareTransformerEncoder',
    'DeformableAttention',
    'MultiScaleDecoder',
    'BoundaryAwareHead'
]
