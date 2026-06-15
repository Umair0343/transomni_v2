"""
Scale-Aware Transformer Blocks for Multi-Scale Feature Processing

Inspired by:
- Deformable DETR (ICLR 2021): Deformable attention for sparse sampling
- SegFormer (NeurIPS 2021): Efficient self-attention with linear complexity
- Swin Transformer (ICCV 2021): Window-based attention
- PVT (ICCV 2021): Pyramid Vision Transformer

Key Features:
1. Scale-Aware Deformable Attention: Adapts sampling points based on object scale
2. Multi-Scale Token Fusion: Combines class and scale information
3. Efficient Attention: Linear complexity for high-resolution features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding"""
    def __init__(self, embed_dim: int, max_h: int = 512, max_w: int = 512):
        super().__init__()
        pe = torch.zeros(embed_dim, max_h, max_w)
        d_model = embed_dim // 2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pos_h = torch.arange(0, max_h).unsqueeze(1)
        pos_w = torch.arange(0, max_w).unsqueeze(1)

        pe[0:d_model:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        pe[1:d_model:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        pe[d_model::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        pe[d_model + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input features"""
        B, C, H, W = x.shape
        return x + self.pe[:, :C, :H, :W]


class DeformableAttention(nn.Module):
    """
    Scale-Aware Deformable Attention

    Adapts the attention sampling pattern based on the scale_id:
    - For large objects (scale 0): Smaller sampling offsets
    - For small objects (scale 3): Larger sampling offsets to capture more context

    This helps the model adapt its receptive field based on object size.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, num_points: int = 4,
                 num_scales: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_scales = num_scales
        self.head_dim = embed_dim // num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Sampling offset prediction
        self.offset_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, num_heads * num_points * 2)  # 2 for x, y offsets
        )

        # Attention weights for each sampling point
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, num_heads * num_points)
        )

        # Scale-specific offset scaling factors
        # Larger scale_id (smaller objects) -> larger offsets for more context
        self.scale_factors = nn.Parameter(torch.tensor([1.0, 1.5, 2.0, 3.0]))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.offset_network[-1].weight, 0)
        nn.init.constant_(self.offset_network[-1].bias, 0)
        nn.init.xavier_uniform_(self.attention_weights[-1].weight)
        nn.init.constant_(self.attention_weights[-1].bias, 0)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                scale_id: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [B, H*W, C]
            key: [B, H*W, C]
            value: [B, H*W, C]
            scale_id: Scale identifier [B]
            reference_points: Optional reference points [B, H*W, 2]
        """
        B, N, C = query.shape
        H = W = int(math.sqrt(N))

        # Get scale factor
        if scale_id is not None:
            scale_idx = scale_id[0].long().item() if scale_id.numel() > 0 else 0
            scale_idx = min(scale_idx, self.num_scales - 1)
            scale_factor = self.scale_factors[scale_idx]
        else:
            scale_factor = 1.0

        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Predict sampling offsets
        offsets = self.offset_network(query)  # B, N, num_heads * num_points * 2
        offsets = offsets.view(B, N, self.num_heads, self.num_points, 2)
        offsets = offsets * scale_factor  # Scale offsets based on object size

        # Predict attention weights
        attn_weights = self.attention_weights(query)  # B, N, num_heads * num_points
        attn_weights = attn_weights.view(B, N, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Generate reference points if not provided
        if reference_points is None:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1, H, device=query.device),
                torch.linspace(0, 1, W, device=query.device),
                indexing='ij'
            )
            reference_points = torch.stack([grid_x, grid_y], dim=-1)  # H, W, 2
            reference_points = reference_points.view(1, N, 1, 1, 2).expand(B, -1, self.num_heads, self.num_points, -1)

        # Sample points = reference + offset
        sampling_points = reference_points + offsets * 0.1  # Scale down offsets
        sampling_points = sampling_points.clamp(0, 1)

        # Reshape value for sampling
        v = v.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)  # B, num_heads, head_dim, H, W

        # Sample values at sampling points using grid_sample
        sampled_values = []
        for h in range(self.num_heads):
            for p in range(self.num_points):
                # Get sampling grid for this head and point
                grid = sampling_points[:, :, h, p, :]  # B, N, 2
                grid = grid.view(B, H, W, 2)
                # Convert to grid_sample format [-1, 1]
                grid = grid * 2 - 1

                # Sample from value
                sampled = F.grid_sample(v[:, h], grid, mode='bilinear',
                                        padding_mode='border', align_corners=False)  # B, head_dim, H, W
                sampled_values.append(sampled)

        # Stack and apply attention weights
        sampled_values = torch.stack(sampled_values, dim=-1)  # B, head_dim, H, W, num_heads*num_points
        sampled_values = sampled_values.view(B, self.head_dim, H, W, self.num_heads, self.num_points)
        sampled_values = sampled_values.permute(0, 4, 2, 3, 1, 5)  # B, num_heads, H, W, head_dim, num_points

        # Apply attention weights
        attn_weights = attn_weights.view(B, H, W, self.num_heads, self.num_points)
        attn_weights = attn_weights.permute(0, 3, 1, 2, 4).unsqueeze(-2)  # B, num_heads, H, W, 1, num_points

        output = (sampled_values * attn_weights).sum(dim=-1)  # B, num_heads, H, W, head_dim
        output = output.permute(0, 2, 3, 1, 4).reshape(B, N, C)

        output = self.out_proj(output)
        output = self.dropout(output)

        return output


class EfficientAttention(nn.Module):
    """
    Efficient Self-Attention with linear complexity.

    Uses spatial reduction to achieve O(N) complexity instead of O(N^2).
    Particularly useful for high-resolution feature maps.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, sr_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Spatial reduction for key and value
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, sr_ratio, sr_ratio, bias=False),
                nn.GroupNorm(min(16, embed_dim), embed_dim)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply spatial reduction to K, V
        if self.sr_ratio > 1:
            x_reshape = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_reshape).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.kv_proj(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv_proj(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)

        return x


class MLP(nn.Module):
    """MLP with GELU activation and dropout"""
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, dropout: float = 0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ScaleAwareTransformerBlock(nn.Module):
    """
    Transformer block with scale-aware attention and class/scale token injection.

    Features:
    - Deformable or efficient attention based on feature resolution
    - Class and scale token modulation
    - Pre-norm architecture for training stability
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, use_deformable: bool = True,
                 num_scales: int = 4, num_classes: int = 6, sr_ratio: int = 4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        if use_deformable:
            self.attn = DeformableAttention(embed_dim, num_heads, num_points=4,
                                            num_scales=num_scales, dropout=dropout)
        else:
            self.attn = EfficientAttention(embed_dim, num_heads, sr_ratio=sr_ratio, dropout=dropout)

        self.use_deformable = use_deformable

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

        # Token injection layers
        self.token_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.token_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: int, W: int,
                cls_token: torch.Tensor, sls_token: torch.Tensor,
                scale_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features [B, H*W, C]
            H, W: Spatial dimensions
            cls_token: Class token [1, C]
            sls_token: Scale token [1, C]
            scale_id: Scale identifier
        """
        B, N, C = x.shape

        # Combine class and scale tokens
        combined_token = torch.cat([cls_token, sls_token], dim=-1)  # 1, 2C
        combined_token = self.token_proj(combined_token)  # 1, C

        # Token-based gating
        gate = self.token_gate(combined_token)  # 1, C
        gate = gate.unsqueeze(1).expand(B, N, -1)  # B, N, C

        # Pre-norm attention with token modulation
        x_norm = self.norm1(x)
        if self.use_deformable:
            attn_out = self.attn(x_norm, x_norm, x_norm, scale_id)
        else:
            attn_out = self.attn(x_norm, H, W)

        # Apply gating and residual
        x = x + self.dropout(attn_out * gate)

        # FFN
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x


class ScaleAwareTransformerEncoder(nn.Module):
    """
    Scale-Aware Transformer Encoder for multi-scale feature processing.

    Processes features from different scales with appropriate attention mechanisms:
    - High resolution (C1, C2): Efficient attention with spatial reduction
    - Low resolution (C3, C4): Deformable attention for precise localization
    """
    def __init__(self, embed_dims: List[int] = [64, 128, 256, 256],
                 num_heads: List[int] = [2, 4, 8, 8],
                 depths: List[int] = [2, 2, 4, 2],
                 num_classes: int = 6, num_scales: int = 4,
                 dropout: float = 0.1, token_dim: int = None,
                 sr_ratios: List[int] = [32, 16]):
        super().__init__()

        self.num_stages = len(embed_dims)
        token_dim = token_dim if token_dim is not None else embed_dims[-1]

        # sr_ratios only apply to EfficientAttention stages (i < 2).
        # With no stem, C1 is at full input resolution (e.g. 512×512) and C2 at half.
        # sr_ratio=32 for C1 reduces K,V to 16×16; sr_ratio=16 for C2 gives 16×16 too.
        # This keeps attention memory manageable at high resolution.

        # Build transformer stages
        self.stages = nn.ModuleList()
        efficient_stage_idx = 0  # counter for efficient-attention stages
        for i in range(self.num_stages):
            use_deformable = (i >= 2)
            if not use_deformable:
                sr = sr_ratios[efficient_stage_idx] if efficient_stage_idx < len(sr_ratios) else 4
                efficient_stage_idx += 1
            else:
                sr = 1  # unused for deformable

            blocks = nn.ModuleList([
                ScaleAwareTransformerBlock(
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=4.0,
                    dropout=dropout,
                    use_deformable=use_deformable,
                    num_scales=num_scales,
                    num_classes=num_classes,
                    sr_ratio=sr
                ) for _ in range(depths[i])
            ])
            self.stages.append(blocks)

        # Positional encoding for each stage
        self.pos_encodings = nn.ModuleList([
            PositionalEncoding2D(dim) for dim in embed_dims
        ])

        # Projection layers from token dimension to each stage dimension
        self.token_projs = nn.ModuleList([
            nn.Linear(token_dim, dim) if dim != token_dim else nn.Identity()
            for dim in embed_dims
        ])

    def forward(self, features: List[torch.Tensor], cls_token: torch.Tensor,
                sls_token: torch.Tensor, scale_id: torch.Tensor) -> List[torch.Tensor]:
        """
        Process multi-scale features through transformer stages.

        Args:
            features: List of CNN features [C1, C2, C3, C4]
            cls_token: Class embedding [1, C]
            sls_token: Scale embedding [1, C]
            scale_id: Scale identifier

        Returns:
            Processed features at each scale
        """
        outputs = []

        for i, (feat, stage, pos_enc, token_proj) in enumerate(
            zip(features, self.stages, self.pos_encodings, self.token_projs)
        ):
            B, C, H, W = feat.shape

            # Add positional encoding
            feat = pos_enc(feat)

            # Flatten to sequence
            feat = feat.flatten(2).transpose(1, 2)  # B, H*W, C

            # Project tokens to current dimension
            cls_proj = token_proj(cls_token)
            sls_proj = token_proj(sls_token)

            # Process through transformer blocks
            for block in stage:
                feat = block(feat, H, W, cls_proj, sls_proj, scale_id)

            # Reshape back to spatial
            feat = feat.transpose(1, 2).reshape(B, C, H, W)
            outputs.append(feat)

        return outputs


if __name__ == '__main__':
    # Test transformer encoder
    embed_dims = [64, 128, 256, 256]

    # Simulate CNN features
    features = [
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16)
    ]

    cls_token = torch.randn(1, 256)
    sls_token = torch.randn(1, 256)
    scale_id = torch.tensor([1])

    encoder = ScaleAwareTransformerEncoder(
        embed_dims=embed_dims,
        num_heads=[2, 4, 8, 8],
        depths=[1, 1, 2, 1],
        num_classes=6,
        num_scales=4
    )

    outputs = encoder(features, cls_token, sls_token, scale_id)
    print("Transformer output shapes:")
    for i, out in enumerate(outputs):
        print(f"  Stage {i+1}: {out.shape}")
