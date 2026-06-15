"""
HybridScaleTransformer: CNN-Transformer Hybrid for Multi-Scale Pathology Segmentation

This model is specifically designed for kidney pathology segmentation with:
- Multiple structure scales (5x, 10x, 40x magnifications)
- Class imbalance (PTC has fewer images but dense annotations)
- Varying object sizes (TUFT/CAP large, PTC small)
- Multiple staining types (H&E, PAS, SIL, TRI)

Architecture Overview:
1. Multi-Scale CNN Encoder: Extracts hierarchical features with scale-aware convolutions
2. Scale-Aware Transformer: Processes features with deformable attention
3. Multi-Scale Decoder: FPN-style fusion with boundary-aware segmentation

Key Innovations:
- Scale-conditioned attention adapts receptive field to object size
- Class/scale tokens modulate feature processing throughout the network
- Boundary-aware heads for precise segmentation of small structures
- Dynamic heads generate task-specific segmentation parameters

Reference Papers:
- TransUNet (CVPR 2021)
- SegFormer (NeurIPS 2021)
- Deformable DETR (ICLR 2021)
- HATs (CVPR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .cnn_encoder import MultiScaleCNNEncoder
from .pretrained_encoder import PretrainedResNetEncoder
from .transformer_blocks import ScaleAwareTransformerEncoder
from .decoder import MultiScaleDecoder


class HybridScaleTransformer(nn.Module):
    """
    Hybrid CNN-Transformer model for multi-scale kidney pathology segmentation.

    Args:
        in_channels: Number of input channels (3 for RGB)
        base_channels: Base channels for CNN encoder (64)
        num_classes: Number of segmentation classes (2 for binary)
        num_task_classes: Number of task/structure types (6: DT, PT, CAP, TUFT, VES, PTC)
        num_scales: Number of scale levels (4: 5x, 10x, 20x, 40x)
        use_transformer: Whether to use transformer encoder (default True)
        transformer_depths: Depths at each transformer stage
        dropout: Dropout rate for regularization

    Input:
        image: [B, 3, H, W] - RGB image patches (512x512)
        task_id: [B] - Task/class identifier (0-5)
        scale_id: [B] - Scale identifier (0-3)

    Output:
        Dictionary containing:
            - 'seg_logits': [B, 2, H, W] - Segmentation logits
            - 'boundary_logits': [B, 1, H, W] - Boundary prediction
    """

    # Class mapping for reference
    CLASS_MAPPING = {
        0: {'name': 'DT', 'full_name': 'Distal Tubular', 'scale': 1, 'mag': '10x'},
        1: {'name': 'PT', 'full_name': 'Proximal Tubular', 'scale': 1, 'mag': '10x'},
        2: {'name': 'CAP', 'full_name': 'Glomerular Unit', 'scale': 0, 'mag': '5x'},
        3: {'name': 'TUFT', 'full_name': 'Glomerular Tuft', 'scale': 0, 'mag': '5x'},
        4: {'name': 'VES', 'full_name': 'Arteries', 'scale': 1, 'mag': '10x'},
        5: {'name': 'PTC', 'full_name': 'Peritubular Capillaries', 'scale': 3, 'mag': '40x'},
    }

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 2,
        num_task_classes: int = 6,
        num_scales: int = 4,
        use_transformer: bool = True,
        transformer_depths: List[int] = [1, 1, 2, 1],
        dropout: float = 0.1,
        backbone: str = 'custom',  # 'custom', 'resnet34', 'resnet50'
        pretrained_backbone: bool = True,
        freeze_backbone_stages: int = 0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_task_classes = num_task_classes
        self.num_scales = num_scales
        self.use_transformer = use_transformer
        self.backbone_type = backbone

        # Feature dimensions at each scale
        encoder_dims = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        # [64, 128, 256, 256]

        # CNN Encoder (custom or pretrained)
        if backbone in ['resnet34', 'resnet50']:
            self.cnn_encoder = PretrainedResNetEncoder(
                backbone=backbone,
                pretrained=pretrained_backbone,
                num_classes=num_task_classes,
                num_scales=num_scales,
                freeze_stages=freeze_backbone_stages
            )
            # Token dimension for pretrained encoder
            token_dim = 512  # PretrainedResNetEncoder uses 512 for embeddings
        else:
            self.cnn_encoder = MultiScaleCNNEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_classes=num_task_classes,
                num_scales=num_scales
            )
            token_dim = base_channels * 8  # Custom encoder token dimension

        # Transformer Encoder (optional but recommended)
        if use_transformer:
            self.transformer_encoder = ScaleAwareTransformerEncoder(
                embed_dims=encoder_dims,
                num_heads=[2, 4, 8, 8],
                depths=transformer_depths,
                num_classes=num_task_classes,
                num_scales=num_scales,
                dropout=dropout,
                token_dim=token_dim  # Match encoder's token dimension
            )

        # Multi-scale Decoder
        self.decoder = MultiScaleDecoder(
            in_channels=encoder_dims,
            decoder_channels=256,
            num_classes=num_classes,
            num_task_classes=num_task_classes,
            num_scales=num_scales
        )

        # Auxiliary classifier for deep supervision (helps with training)
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(encoder_dims[-1], encoder_dims[-1] // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16, encoder_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(encoder_dims[-1] // 2, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        image: torch.Tensor,
        task_id: torch.Tensor,
        scale_id: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hybrid model.

        Args:
            image: Input image [B, 3, H, W]
            task_id: Task/class identifier [B]
            scale_id: Scale identifier [B]

        Returns:
            Dictionary with segmentation and boundary predictions
        """
        B, C, H, W = image.shape

        # Normalize input to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

        # CNN Encoding
        features, cls_token, sls_token = self.cnn_encoder(image, task_id, scale_id)
        # features: [C1, C2, C3, C4] at 1/4, 1/8, 1/16, 1/32 resolution

        # Transformer Processing (optional)
        if self.use_transformer:
            features = self.transformer_encoder(features, cls_token, sls_token, scale_id)

        # Decoding
        outputs = self.decoder(features, cls_token, sls_token, task_id, scale_id)

        # Auxiliary output for deep supervision
        aux_logits = self.aux_classifier(features[-1])
        aux_logits = F.interpolate(aux_logits, size=(H, W), mode='bilinear', align_corners=False)
        outputs['aux_logits'] = aux_logits

        # Ensure output matches input resolution
        if outputs['seg_logits'].shape[2:] != (H, W):
            outputs['seg_logits'] = F.interpolate(
                outputs['seg_logits'], size=(H, W), mode='bilinear', align_corners=False
            )
        if outputs['boundary_logits'].shape[2:] != (H, W):
            outputs['boundary_logits'] = F.interpolate(
                outputs['boundary_logits'], size=(H, W), mode='bilinear', align_corners=False
            )

        return outputs

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_class_info(self, task_id: int) -> dict:
        """Get information about a specific class"""
        return self.CLASS_MAPPING.get(task_id, {'name': 'Unknown', 'scale': 0})


def build_hybrid_model(
    num_task_classes: int = 6,
    num_scales: int = 4,
    use_transformer: bool = True,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    backbone: str = 'custom',
    pretrained_backbone: bool = True,
    freeze_backbone_stages: int = 0,
    dropout: float = 0.1
) -> HybridScaleTransformer:
    """
    Build the hybrid CNN-Transformer model.

    Args:
        num_task_classes: Number of task/structure types (default 6)
        num_scales: Number of scale levels (default 4)
        use_transformer: Whether to use transformer encoder
        pretrained: Whether to load pretrained weights from checkpoint
        checkpoint_path: Path to checkpoint file
        backbone: Backbone type ('custom', 'resnet34', 'resnet50')
        pretrained_backbone: Whether to use ImageNet pretrained backbone
        freeze_backbone_stages: Number of backbone stages to freeze (0-4)
        dropout: Dropout rate for regularization

    Returns:
        HybridScaleTransformer model
    """
    model = HybridScaleTransformer(
        in_channels=3,
        base_channels=64,
        num_classes=2,  # Binary segmentation
        num_task_classes=num_task_classes,
        num_scales=num_scales,
        use_transformer=use_transformer,
        transformer_depths=[1, 1, 2, 1],
        dropout=dropout,
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
        freeze_backbone_stages=freeze_backbone_stages
    )

    if pretrained and checkpoint_path is not None:
        print(f"Loading pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)

    return model


class HybridScaleTransformerLite(HybridScaleTransformer):
    """
    Lightweight version of HybridScaleTransformer for faster inference.

    Reduces model size while maintaining performance for deployment.
    """
    def __init__(
        self,
        num_task_classes: int = 6,
        num_scales: int = 4,
        dropout: float = 0.05
    ):
        super().__init__(
            in_channels=3,
            base_channels=32,  # Reduced from 64
            num_classes=2,
            num_task_classes=num_task_classes,
            num_scales=num_scales,
            use_transformer=False,  # Skip transformer for speed
            transformer_depths=[1, 1, 1, 1],
            dropout=dropout
        )


if __name__ == '__main__':
    # Test model
    print("=" * 60)
    print("Testing HybridScaleTransformer")
    print("=" * 60)

    model = build_hybrid_model(
        num_task_classes=6,
        num_scales=4,
        use_transformer=True
    )

    print(f"\nTotal parameters: {model.get_num_parameters():,}")

    # Test forward pass
    batch_size = 2
    image = torch.randn(batch_size, 3, 512, 512)
    task_id = torch.tensor([0])  # DT
    scale_id = torch.tensor([1])  # 10x

    print(f"\nInput shape: {image.shape}")
    print(f"Task ID: {task_id.item()} ({model.get_class_info(task_id.item())['name']})")
    print(f"Scale ID: {scale_id.item()}")

    with torch.no_grad():
        outputs = model(image, task_id, scale_id)

    print(f"\nOutputs:")
    print(f"  Segmentation logits: {outputs['seg_logits'].shape}")
    print(f"  Boundary logits: {outputs['boundary_logits'].shape}")
    print(f"  Auxiliary logits: {outputs['aux_logits'].shape}")

    # Test all classes
    print("\n" + "=" * 60)
    print("Testing all task classes")
    print("=" * 60)
    for task_id_val in range(6):
        info = model.get_class_info(task_id_val)
        scale_id_val = info['scale']
        task_id = torch.tensor([task_id_val])
        scale_id = torch.tensor([scale_id_val])

        with torch.no_grad():
            outputs = model(image, task_id, scale_id)

        print(f"Task {task_id_val} ({info['name']:4s}, {info['mag']:3s}): "
              f"output shape = {outputs['seg_logits'].shape}")

    print("\nModel test passed!")
