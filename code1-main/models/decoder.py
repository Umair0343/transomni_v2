"""
Multi-Scale Decoder with Boundary-Aware Segmentation Head

Inspired by:
- UNet++ (DLMIA 2018): Dense skip connections
- Boundary-Aware Networks (MICCAI): Explicit boundary supervision
- SegFormer Decoder (NeurIPS 2021): Lightweight MLP decoder
- Dynamic Head (CVPR 2021): Instance-specific heads

Key Features:
1. Multi-scale feature fusion with boundary enhancement
2. Dynamic segmentation head conditioned on class and scale
3. Boundary-aware refinement for precise segmentation
4. Auxiliary boundary supervision for better edge detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """Basic convolution block with GroupNorm and ReLU"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(min(16, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class ChannelAttention(nn.Module):
    """Channel attention for feature recalibration"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        out = avg_out + max_out
        return x * out.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention for focusing on important regions"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class BoundaryDetector(nn.Module):
    """
    Explicit boundary detection module.

    Uses edge detection principles to identify object boundaries,
    particularly important for small structures like PTC.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Laplacian-like edge detection
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )

        # Sobel-like gradient detection
        self.gradient_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )

        # Initialize edge conv with Laplacian kernel
        self._init_edge_kernels()

    def _init_edge_kernels(self):
        # Laplacian kernel approximation
        laplacian = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32)

        with torch.no_grad():
            for i in range(self.edge_conv[0].weight.shape[0]):
                self.edge_conv[0].weight[i, 0] = laplacian

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self.edge_conv(x)
        gradient = self.gradient_conv(x)
        boundary = self.fusion(torch.cat([edge, gradient], dim=1))
        return boundary


class BoundaryAwareHead(nn.Module):
    """
    Boundary-aware segmentation head with auxiliary boundary prediction.

    Provides both segmentation output and explicit boundary map for
    multi-task learning.
    """
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()

        self.boundary_detector = BoundaryDetector(in_channels, in_channels // 2)

        # Segmentation branch
        self.seg_conv = nn.Sequential(
            ConvBlock(in_channels + in_channels // 2, in_channels),
            ConvBlock(in_channels, in_channels // 2),
            nn.Conv2d(in_channels // 2, num_classes, 1)
        )

        # Boundary prediction branch
        self.boundary_conv = nn.Sequential(
            ConvBlock(in_channels // 2, in_channels // 4),
            nn.Conv2d(in_channels // 4, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            seg_logits: Segmentation logits [B, num_classes, H, W]
            boundary_logits: Boundary map [B, 1, H, W]
        """
        boundary_feat = self.boundary_detector(x)

        # Segmentation with boundary awareness
        seg_input = torch.cat([x, boundary_feat], dim=1)
        seg_logits = self.seg_conv(seg_input)

        # Boundary prediction
        boundary_logits = self.boundary_conv(boundary_feat)

        return seg_logits, boundary_logits


class FeaturePyramidFusion(nn.Module):
    """
    Feature Pyramid Network (FPN) style multi-scale feature fusion.

    Combines features from different scales with top-down pathway
    and lateral connections.
    """
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()

        self.num_levels = len(in_channels)

        # Lateral connections (1x1 convs)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, bias=False)
            for c in in_channels
        ])

        # Output convs (3x3 convs)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(min(16, out_channels), out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels
        ])

        # Attention modules
        self.attentions = nn.ModuleList([
            CBAM(out_channels) for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: Multi-scale features from encoder [C1, C2, C3, C4]
                     from high resolution to low resolution

        Returns:
            Fused features at each scale
        """
        # Build top-down pathway
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down fusion (from low res to high res)
        for i in range(self.num_levels - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=(h, w), mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output convs and attention
        outputs = []
        for i, (lateral, output_conv, attn) in enumerate(
            zip(laterals, self.output_convs, self.attentions)
        ):
            out = output_conv(lateral)
            out = attn(out)
            outputs.append(out)

        return outputs


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder with progressive upsampling and feature fusion.

    Combines FPN-style fusion with UNet-style skip connections
    and boundary-aware refinement.
    """
    def __init__(self, in_channels: List[int] = [64, 128, 256, 256],
                 decoder_channels: int = 256, num_classes: int = 2,
                 num_task_classes: int = 6, num_scales: int = 4):
        super().__init__()

        self.num_classes = num_classes
        self.num_task_classes = num_task_classes
        self.num_scales = num_scales

        # Feature pyramid fusion
        self.fpn = FeaturePyramidFusion(in_channels, decoder_channels)

        # Progressive upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.up_blocks.append(nn.Sequential(
                ConvBlock(decoder_channels * 2, decoder_channels),
                ConvBlock(decoder_channels, decoder_channels),
                CBAM(decoder_channels)
            ))

        # Final refinement (no upsampling).
        # With the stem removed, C1 — and therefore the fused decoder feature after
        # the progressive-upsampling loop — is already at full input resolution.
        # The previous ×4 Upsample here was sized for the old stem (C1 at 1/4 res)
        # and would overshoot to 4× the input, wasting compute and risking OOM.
        self.final_up = nn.Sequential(
            ConvBlock(decoder_channels, decoder_channels // 2),
            ConvBlock(decoder_channels // 2, decoder_channels // 4)
        )

        # Pre-head feature processing
        self.pre_head = nn.Sequential(
            ConvBlock(decoder_channels // 4, 32),
            nn.Conv2d(32, 8, 1)
        )

        # Dynamic head parameters generation
        # 162 = 8*8 + 8*8 + 8*2 + 8 + 8 + 2 (weights and biases for 3-layer head)
        embed_dim = in_channels[-1]
        self.head_controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 162)
        )

        # Class and scale conditioning
        self.cls_emb = nn.Parameter(torch.randn(1, num_task_classes, 64))
        self.sls_emb = nn.Parameter(torch.randn(1, num_scales, 64))

        self.condition_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 162)
        )

        # Boundary-aware head (for auxiliary loss)
        self.boundary_head = BoundaryAwareHead(decoder_channels // 4, num_classes)

    def parse_dynamic_params(self, params: torch.Tensor, channels: int = 8):
        """Parse dynamic convolution parameters"""
        weight_nums = [8 * 8, 8 * 8, 8 * 2]  # 64, 64, 16
        bias_nums = [8, 8, 2]

        num_insts = params.size(0)
        splits = list(torch.split(params, weight_nums + bias_nums, dim=1))

        weight_splits = splits[:3]
        bias_splits = splits[3:]

        # Reshape weights and biases
        weight_splits[0] = weight_splits[0].reshape(num_insts * channels, channels, 1, 1)
        weight_splits[1] = weight_splits[1].reshape(num_insts * channels, channels, 1, 1)
        weight_splits[2] = weight_splits[2].reshape(num_insts * 2, channels, 1, 1)

        bias_splits[0] = bias_splits[0].reshape(num_insts * channels)
        bias_splits[1] = bias_splits[1].reshape(num_insts * channels)
        bias_splits[2] = bias_splits[2].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def dynamic_head_forward(self, features: torch.Tensor,
                             weights: List[torch.Tensor],
                             biases: List[torch.Tensor],
                             num_insts: int) -> torch.Tensor:
        """Apply dynamic convolution head"""
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < len(weights) - 1:
                x = F.relu(x)
        return x

    def forward(self, features: List[torch.Tensor],
                cls_token: torch.Tensor, sls_token: torch.Tensor,
                task_id: torch.Tensor, scale_id: torch.Tensor) -> dict:
        """
        Args:
            features: Multi-scale features [C1, C2, C3, C4]
            cls_token: Class token from encoder
            sls_token: Scale token from encoder
            task_id: Task identifier
            scale_id: Scale identifier

        Returns:
            Dictionary with:
                - 'seg_logits': Main segmentation output [B, 2, H, W]
                - 'boundary_logits': Boundary map [B, 1, H, W]
                - 'aux_logits': Auxiliary outputs for deep supervision
        """
        B = features[0].shape[0]

        # FPN fusion
        fpn_features = self.fpn(features)  # [P1, P2, P3, P4]

        # Progressive upsampling and fusion (from P4 to P1)
        x = fpn_features[-1]  # Start from lowest resolution
        for i in range(len(self.up_blocks) - 1, -1, -1):
            # Upsample
            h, w = fpn_features[i].shape[2:]
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            # Concatenate with lateral feature
            x = torch.cat([x, fpn_features[i]], dim=1)
            # Process
            x = self.up_blocks[i](x)

        # Final upsampling
        x = self.final_up(x)

        # Get conditioning
        task_idx = task_id[0].long().item() if task_id.numel() > 0 else 0
        scale_idx = scale_id[0].long().item() if scale_id.numel() > 0 else 0
        task_idx = min(task_idx, self.num_task_classes - 1)
        scale_idx = min(scale_idx, self.num_scales - 1)

        cls_cond = self.cls_emb[:, task_idx, :]  # 1, 64
        sls_cond = self.sls_emb[:, scale_idx, :]  # 1, 64
        combined_cond = torch.cat([cls_cond, sls_cond], dim=-1)  # 1, 128
        cond_params = self.condition_proj(combined_cond)  # 1, 162

        # Generate dynamic head parameters
        head_params = self.head_controller(fpn_features[-1])  # B, 162
        head_params = head_params + cond_params.expand(B, -1)  # Add conditioning

        # Pre-head features
        head_input = self.pre_head(x)  # B, 8, H, W
        N, C, H, W = head_input.shape
        head_input_flat = head_input.reshape(1, -1, H, W)

        # Apply dynamic head
        weights, biases = self.parse_dynamic_params(head_params)
        seg_logits = self.dynamic_head_forward(head_input_flat, weights, biases, N)
        seg_logits = seg_logits.reshape(N, 2, H, W)

        # Boundary prediction
        _, boundary_logits = self.boundary_head(x)

        return {
            'seg_logits': seg_logits,
            'boundary_logits': boundary_logits,
            'features': x  # For auxiliary supervision if needed
        }


if __name__ == '__main__':
    # Test decoder
    in_channels = [64, 128, 256, 256]
    decoder = MultiScaleDecoder(
        in_channels=in_channels,
        decoder_channels=256,
        num_classes=2,
        num_task_classes=6,
        num_scales=4
    )

    # Simulate features
    features = [
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16)
    ]

    cls_token = torch.randn(1, 256)
    sls_token = torch.randn(1, 256)
    task_id = torch.tensor([0])
    scale_id = torch.tensor([1])

    outputs = decoder(features, cls_token, sls_token, task_id, scale_id)
    print("Decoder outputs:")
    print(f"  Segmentation logits: {outputs['seg_logits'].shape}")
    print(f"  Boundary logits: {outputs['boundary_logits'].shape}")
