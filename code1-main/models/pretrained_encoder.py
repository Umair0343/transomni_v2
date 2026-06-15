"""
Pretrained ResNet Encoder for Multi-Scale Pathology Segmentation

Uses ImageNet pretrained weights for better feature extraction.
Pretrained backbones significantly improve generalization, especially
for medical imaging with limited training data.

Features:
- ResNet34/50 backbone with pretrained ImageNet weights
- Maintains compatibility with existing transformer and decoder
- Scale-conditioned processing for multi-magnification input
- Proper normalization for pretrained models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import torchvision.models as models


class PretrainedResNetEncoder(nn.Module):
    """
    Pretrained ResNet encoder with scale-aware conditioning.

    Uses ImageNet pretrained weights and adapts them for pathology segmentation.

    Outputs features at 4 scales (same as original encoder):
    - C1: 1/4 resolution, 64 channels
    - C2: 1/8 resolution, 128 channels
    - C3: 1/16 resolution, 256 channels
    - C4: 1/32 resolution, 256 channels (reduced from 512 for decoder compatibility)

    Args:
        backbone: ResNet variant ('resnet34' or 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        num_classes: Number of task classes (6)
        num_scales: Number of scale levels (4)
        freeze_bn: Whether to freeze batch norm layers
        freeze_stages: Number of initial stages to freeze (0-4)
    """

    # ImageNet normalization (for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        backbone: str = 'resnet34',
        pretrained: bool = True,
        num_classes: int = 6,
        num_scales: int = 4,
        freeze_bn: bool = False,
        freeze_stages: int = 0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_scales = num_scales
        self.freeze_bn = freeze_bn

        # Load pretrained backbone
        if backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.expansion = 1
            stage_channels = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.expansion = 4
            stage_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract backbone components
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64*expansion
        self.layer2 = resnet.layer2  # 1/8, 128*expansion
        self.layer3 = resnet.layer3  # 1/16, 256*expansion
        self.layer4 = resnet.layer4  # 1/32, 512*expansion

        # Channel adaptation to match original encoder output dimensions
        # Original: [64, 128, 256, 256]
        self.adapt1 = nn.Conv2d(stage_channels[0], 64, 1, bias=False)
        self.adapt2 = nn.Conv2d(stage_channels[1], 128, 1, bias=False)
        self.adapt3 = nn.Conv2d(stage_channels[2], 256, 1, bias=False)
        self.adapt4 = nn.Conv2d(stage_channels[3], 256, 1, bias=False)

        # Scale-conditioned refinement
        self.scale_conv = ScaleConditionedConvPretrained(256, 256, num_scales)

        # ASPP for multi-scale context
        self.aspp = ASPPPretrained(256, 256)

        # Class and scale embeddings (same dimension as original: base_channels * 8 = 512)
        embed_dim = 512
        self.embed_dim = embed_dim
        self.cls_emb = nn.Parameter(torch.randn(1, num_classes, embed_dim))
        self.sls_emb = nn.Parameter(torch.randn(1, num_scales, embed_dim))

        # Project ASPP features to embed_dim for modulation
        self.feature_proj = nn.Conv2d(256, embed_dim, 1, bias=False)
        self.feature_proj_back = nn.Conv2d(embed_dim, 256, 1, bias=False)

        # Class/scale modulation (FiLM-style)
        self.class_modulation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim * 2)  # gamma and beta
        )

        # Register ImageNet normalization
        self.register_buffer('mean', torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        # Freeze stages if requested
        self._freeze_stages(freeze_stages)

        # Initialize new layers
        self._init_new_layers()

    def _freeze_stages(self, num_stages: int):
        """Freeze early stages of the backbone"""
        if num_stages >= 1:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
        if num_stages >= 2:
            for param in self.layer1.parameters():
                param.requires_grad = False
        if num_stages >= 3:
            for param in self.layer2.parameters():
                param.requires_grad = False
        if num_stages >= 4:
            for param in self.layer3.parameters():
                param.requires_grad = False

    def _init_new_layers(self):
        """Initialize newly added layers"""
        for m in [self.adapt1, self.adapt2, self.adapt3, self.adapt4,
                  self.feature_proj, self.feature_proj_back]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        nn.init.trunc_normal_(self.cls_emb, std=0.02)
        nn.init.trunc_normal_(self.sls_emb, std=0.02)

    def train(self, mode: bool = True):
        """Override train to optionally freeze batch norm"""
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using ImageNet statistics"""
        # Ensure input is in [0, 1] range
        if x.max() > 1.0:
            x = x / 255.0
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        return x

    def forward(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
        scale_id: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image tensor [B, 3, H, W]
            task_id: Class/task identifier [B]
            scale_id: Scale identifier [B]

        Returns:
            features: List of multi-scale features [C1, C2, C3, C4]
            cls_token: Class embedding
            sls_token: Scale embedding
        """
        B = x.shape[0]

        # Normalize input for pretrained model
        x = self.normalize_input(x)

        # Get class and scale tokens
        task_idx = task_id[0].long().item() if task_id.numel() > 0 else 0
        scale_idx = scale_id[0].long().item() if scale_id.numel() > 0 else 0
        task_idx = min(task_idx, self.num_classes - 1)
        scale_idx = min(scale_idx, self.num_scales - 1)

        cls_token = self.cls_emb[:, task_idx, :]  # 1, embed_dim
        sls_token = self.sls_emb[:, scale_idx, :]  # 1, embed_dim

        # Backbone forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1/4

        c1 = self.layer1(x)   # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32

        # Adapt channels to match original encoder
        c1 = self.adapt1(c1)  # -> 64
        c2 = self.adapt2(c2)  # -> 128
        c3 = self.adapt3(c3)  # -> 256
        c4 = self.adapt4(c4)  # -> 256

        # Scale-conditioned processing
        c4 = self.scale_conv(c4, scale_id)

        # ASPP for multi-scale context
        c4 = self.aspp(c4)

        # Class/scale modulation (FiLM-style)
        combined_token = cls_token + sls_token  # 1, embed_dim
        modulation = self.class_modulation(combined_token)  # 1, embed_dim * 2
        gamma, beta = modulation.chunk(2, dim=-1)

        # Project features to embed_dim for modulation
        c4_proj = self.feature_proj(c4)  # B, embed_dim, H, W
        gamma = gamma.view(1, -1, 1, 1)
        beta = beta.view(1, -1, 1, 1)
        c4_mod = gamma * c4_proj + beta

        # Project back to 256 channels
        c4_out = self.feature_proj_back(c4_mod)

        return [c1, c2, c3, c4_out], cls_token, sls_token


class ScaleConditionedConvPretrained(nn.Module):
    """Scale-conditioned convolution adapted for pretrained encoder"""

    def __init__(self, in_channels: int, out_channels: int, num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales

        # Different dilation rates for different scales
        dilations = [1, 2, 3, 6]

        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, d, dilation=d, bias=False),
                nn.GroupNorm(min(16, out_channels), out_channels),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])

        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, scale_id: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Compute all scale features
        scale_features = [conv(x) for conv in self.scale_convs]
        scale_features = torch.stack(scale_features, dim=1)  # B, num_scales, C, H, W

        # Use scale_id to select primary scale
        primary_scale = scale_id[0].long().item() if scale_id.numel() > 0 else 0
        primary_scale = min(primary_scale, self.num_scales - 1)

        # Soft attention over scales
        scale_weights = self.scale_attention(x)  # B, num_scales

        # Bias towards primary scale
        scale_bias = torch.zeros_like(scale_weights)
        scale_bias[:, primary_scale] = 0.5
        scale_weights = scale_weights + scale_bias
        scale_weights = scale_weights / scale_weights.sum(dim=-1, keepdim=True)

        # Weighted combination
        scale_weights = scale_weights.view(B, self.num_scales, 1, 1, 1)
        out = (scale_features * scale_weights).sum(dim=1)

        return out


class ASPPPretrained(nn.Module):
    """ASPP module for pretrained encoder"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        dilations = [1, 6, 12, 18]

        self.convs = nn.ModuleList()
        # 1x1 conv
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True)
        ))
        # 3x3 convs with different dilations
        for d in dilations[1:]:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(min(16, out_channels), out_channels),
                nn.ReLU(inplace=True)
            ))
        # Global average pooling
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)  # Increased dropout for regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for conv in self.convs:
            features.append(conv(x))

        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        features.append(gap)

        out = torch.cat(features, dim=1)
        return self.fusion(out)


if __name__ == '__main__':
    # Test pretrained encoder
    print("Testing PretrainedResNetEncoder...")

    encoder = PretrainedResNetEncoder(
        backbone='resnet34',
        pretrained=True,
        num_classes=6,
        num_scales=4,
        freeze_stages=0
    )

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    task_id = torch.tensor([0])
    scale_id = torch.tensor([1])

    features, cls_token, sls_token = encoder(x, task_id, scale_id)

    print("\nFeature shapes:")
    for i, f in enumerate(features):
        print(f"  C{i+1}: {f.shape}")
    print(f"Class token: {cls_token.shape}")
    print(f"Scale token: {sls_token.shape}")

    print("\nPretrained encoder test passed!")
