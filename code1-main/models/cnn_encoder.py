"""
Multi-Scale CNN Encoder with Feature Pyramid Network

Inspired by:
- ResNet with dilated convolutions
- Feature Pyramid Networks (FPN)
- Atrous Spatial Pyramid Pooling (ASPP) from DeepLab
- PRPSeg (CVPR 2024): consistent per-stage task conditioning

Features:
- No stem: stages start directly from the raw image, bottleneck stays at H/8
- FiLM conditioning after every encoder stage, not just the last
- ASPP at the bottleneck for multi-scale context
- Group normalization for small batch sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBNReLU(nn.Module):
    """Convolution + GroupNorm + ReLU block"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1,
                 groups: int = 1, num_groups: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=False)
        self.norm = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with optional scale-aware modulation"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dilation: int = 1, num_groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, dilation, dilation, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on combined class+scale token.

    Applied after every encoder stage so that class/scale information shapes
    features at all resolutions, not just the bottleneck.

    Initialized so gamma=1, beta=0 (identity) at the start of training,
    meaning the conditioning is additive and cannot hurt at initialization.
    """
    def __init__(self, token_dim: int, feature_channels: int):
        super().__init__()
        self.proj = nn.Linear(token_dim, feature_channels * 2)
        # Identity init: gamma starts at 1, beta starts at 0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:feature_channels] = 1.0

    def forward(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [B, C, H, W]
            token: [1, token_dim]  — combined cls + sls token
        """
        params = self.proj(token)                    # [1, C*2]
        gamma, beta = params.chunk(2, dim=-1)        # each [1, C]
        return gamma.view(1, -1, 1, 1) * x + beta.view(1, -1, 1, 1)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for multi-scale context.
    Adapted for pathology images with different magnifications.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Dilations tuned for a 64×64 bottleneck feature map.
        # Original DeepLab values [6,12,18] overshoot the map boundaries at this resolution.
        # [1,2,4,6] give useful receptive field variety without going out of bounds.
        dilations = [1, 2, 4, 6]

        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True)
        ))
        for d in dilations[1:]:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(min(16, out_channels), out_channels),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [conv(x) for conv in self.convs]
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        features.append(gap)
        return self.fusion(torch.cat(features, dim=1))


class MultiScaleCNNEncoder(nn.Module):
    """
    Multi-scale CNN encoder with hierarchical feature extraction.

    Key changes vs original:
    - No stem: stage1 receives raw RGB directly (stride=1), bottleneck is H/8.
    - FiLM conditioning after every stage (C1–C4), not just the bottleneck.
      Early features are class-aware from the start, matching PRPSeg's approach.
    - ScaleConditionedConv removed — per-stage FiLM handles this more cleanly.

    Outputs features at 4 scales (for 512×512 input):
    - C1: full resolution  512×512, 64  channels
    - C2: 1/2  resolution  256×256, 128 channels
    - C3: 1/4  resolution  128×128, 256 channels
    - C4: 1/8  resolution   64×64,  256 channels  (after ASPP)
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_classes: int = 6, num_scales: int = 4):
        super().__init__()

        self.num_classes = num_classes
        self.num_scales = num_scales

        # No stem — stages start directly from the raw image.
        # Without early downsampling the bottleneck is 64×64 (input/8)
        # instead of 16×16 (input/32), giving 16× more spatial positions
        # for the decoder and preserving fine structure (e.g. PTC capillaries).
        #
        # Spatial progression:
        #   Stage1 (stride=1): H×W      → C1  [64,  H,   W  ]
        #   Stage2 (stride=2): H/2×W/2  → C2  [128, H/2, W/2]
        #   Stage3 (stride=2): H/4×W/4  → C3  [256, H/4, W/4]
        #   Stage4 (stride=2): H/8×W/8  → C4  [256, H/8, W/8]  (after ASPP)

        # Encoder stages — stage1 takes raw in_channels (3) directly
        self.stage1 = self._make_stage(in_channels,        base_channels,     2, stride=1)
        self.stage2 = self._make_stage(base_channels,      base_channels * 2, 2, stride=2)
        self.stage3 = self._make_stage(base_channels * 2,  base_channels * 4, 2, stride=2)
        self.stage4 = self._make_stage(base_channels * 4,  base_channels * 8, 2, stride=2)

        # ASPP at bottleneck for multi-scale context
        self.aspp = ASPP(base_channels * 8, base_channels * 4)

        # Class and scale token embeddings
        embed_dim = base_channels * 8  # 512
        self.cls_emb = nn.Parameter(torch.randn(1, num_classes, embed_dim))
        self.sls_emb = nn.Parameter(torch.randn(1, num_scales,  embed_dim))

        # Per-stage FiLM modules — conditioning at every resolution
        stage_channels = [
            base_channels,         # after stage1: 64
            base_channels * 2,     # after stage2: 128
            base_channels * 4,     # after stage3: 256
            base_channels * 8,     # after stage4: 512  (before ASPP)
        ]
        self.film_layers = nn.ModuleList([
            FiLM(embed_dim, ch) for ch in stage_channels
        ])

        self._init_weights()

    def _make_stage(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, task_id: torch.Tensor,
                scale_id: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            x:        Input image [B, 3, H, W]
            task_id:  Class/task identifier [B]
            scale_id: Scale identifier [B]

        Returns:
            features:  [C1, C2, C3, C4] at H, H/2, H/4, H/8 resolution
            cls_token: [1, embed_dim]
            sls_token: [1, embed_dim]
        """
        task_idx  = min(task_id[0].long().item()  if task_id.numel()  > 0 else 0, self.num_classes - 1)
        scale_idx = min(scale_id[0].long().item() if scale_id.numel() > 0 else 0, self.num_scales  - 1)

        cls_token = self.cls_emb[:, task_idx,  :]   # [1, embed_dim]
        sls_token = self.sls_emb[:, scale_idx, :]   # [1, embed_dim]
        token     = cls_token + sls_token            # [1, embed_dim]

        # Stage 1 → FiLM → C1  (no stem: full input resolution)
        c1 = self.stage1(x)
        c1 = self.film_layers[0](c1, token)          # [B, 64,  H/4,  W/4]

        # Stage 2 → FiLM → C2
        c2 = self.stage2(c1)
        c2 = self.film_layers[1](c2, token)          # [B, 128, H/8,  W/8]

        # Stage 3 → FiLM → C3
        c3 = self.stage3(c2)
        c3 = self.film_layers[2](c3, token)          # [B, 256, H/16, W/16]

        # Stage 4 → FiLM → ASPP → C4
        c4 = self.stage4(c3)
        c4 = self.film_layers[3](c4, token)          # [B, 512, H/32, W/32]
        c4 = self.aspp(c4)                           # [B, 256, H/32, W/32]

        return [c1, c2, c3, c4], cls_token, sls_token


if __name__ == '__main__':
    model = MultiScaleCNNEncoder(in_channels=3, base_channels=64, num_classes=6, num_scales=4)
    x = torch.randn(2, 3, 512, 512)
    task_id  = torch.tensor([0])
    scale_id = torch.tensor([1])

    features, cls_token, sls_token = model(x, task_id, scale_id)
    print("Feature shapes:")
    for i, f in enumerate(features):
        print(f"  C{i+1}: {f.shape}")
    print(f"Class token: {cls_token.shape}")
    print(f"Scale token: {sls_token.shape}")
