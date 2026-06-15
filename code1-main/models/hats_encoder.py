"""
HATs-Style CNN Encoder with Layer-wise Token Injection + Transformer

Implements the key innovations:
- Class tokens and scale tokens are ADDED at EVERY encoder layer
- Token embeddings are sliced per layer to match channel dimensions
- Transformer at bottleneck for global context
- Dynamic head for task/scale-specific segmentation

Reference: HATs (CVPR 2024) + Transformer enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class PositionalEncoding2D(nn.Module):
    """2D Positional encoding for spatial features"""

    def __init__(self, embed_dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        # Create positional encoding
        pe = torch.zeros(embed_dim, max_h, max_w)

        d_model = embed_dim // 2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pos_h = torch.arange(max_h).unsqueeze(1).float()
        pos_w = torch.arange(max_w).unsqueeze(1).float()

        pe[0:d_model:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)
        pe[1:d_model:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)
        pe[d_model::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)
        pe[d_model + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        return x + self.pe[:, :C, :H, :W]


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head self-attention.

    Used at the bottleneck for global context modeling.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C]"""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class BottleneckTransformer(nn.Module):
    """
    Transformer module at the bottleneck for global context.

    Takes spatial features, flattens to sequence, applies transformer,
    then reshapes back to spatial.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.pos_encoding = PositionalEncoding2D(embed_dim, max_h=64, max_w=64)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial features

        Returns:
            [B, C, H, W] transformed features
        """
        B, C, H, W = x.shape

        # Add positional encoding
        x = self.pos_encoding(x)

        # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Reshape back to spatial: [B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class NoBottleneck(nn.Module):
    """Residual block with GroupNorm (same as original HATs)"""

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 dilation: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.gn1 = nn.GroupNorm(16, inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.gn2 = nn.GroupNorm(16, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, dilation=1, bias=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out


class HATsEncoder(nn.Module):
    """
    HATs-style CNN Encoder with layer-wise class/scale token injection.

    Key difference from our previous implementation:
    - Tokens are ADDED to features at EVERY encoder layer
    - Token dimension = sum of all layer channels (32+32+64+128+256+256=768)
    - Each layer gets a different slice of the token embedding

    This provides strong class/scale conditioning throughout the network,
    which is critical for multi-task learning with varying object sizes.

    Args:
        in_channels: Input channels (3 for RGB)
        num_classes: Number of task classes (6 for kidney pathology)
        num_scales: Number of scale levels (4)
        layers: Number of blocks per stage [1, 2, 2, 2, 2]
    """

    # Channel dimensions at each layer
    LAYER_CHANNELS = [32, 32, 64, 128, 256, 256]  # layer0-4 + head
    TOKEN_DIM = sum(LAYER_CHANNELS)  # 768

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        num_scales: int = 4,
        layers: List[int] = [1, 2, 2, 2, 2]
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_scales = num_scales

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)

        # Encoder stages (matching original HATs structure)
        self.layer0 = self._make_layer(32, 32, layers[0], stride=1)
        self.layer1 = self._make_layer(32, 64, layers[1], stride=2)
        self.layer2 = self._make_layer(64, 128, layers[2], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[3], stride=2)
        self.layer4 = self._make_layer(256, 256, layers[4], stride=2)

        # Fusion convolution at bottleneck
        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Global Average Pooling for dynamic head
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Class and Scale token embeddings
        # Total dimension = 32+32+64+128+256+256 = 768
        self.cls_emb = nn.Parameter(torch.randn(1, num_classes, self.TOKEN_DIM))
        self.sls_emb = nn.Parameter(torch.randn(1, num_scales, self.TOKEN_DIM))

        # Compute slice indices for each layer
        self._compute_slice_indices()

        self._init_weights()

    def _compute_slice_indices(self):
        """Compute start/end indices for token slicing per layer"""
        self.slice_indices = []
        start = 0
        for ch in self.LAYER_CHANNELS:
            end = start + ch
            self.slice_indices.append((start, end))
            start = end

    def _make_layer(self, inplanes: int, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                          padding=0, bias=False)
            )

        layers = [NoBottleneck(inplanes, planes, stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(NoBottleneck(planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize token embeddings
        nn.init.trunc_normal_(self.cls_emb, std=0.02)
        nn.init.trunc_normal_(self.sls_emb, std=0.02)

    def _get_token_slice(self, task_id: torch.Tensor, scale_id: torch.Tensor,
                         layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the token slice for a specific layer.

        Args:
            task_id: Task identifier [B]
            scale_id: Scale identifier [B]
            layer_idx: Layer index (0-5)

        Returns:
            cls_token: Class token slice [B, C, 1, 1]
            sls_token: Scale token slice [B, C, 1, 1]
        """
        B = task_id.shape[0] if task_id.dim() > 0 else 1

        # Get task and scale indices
        task_idx = task_id[0].long().item() if task_id.numel() > 0 else 0
        scale_idx = scale_id[0].long().item() if scale_id.numel() > 0 else 0
        task_idx = min(task_idx, self.num_classes - 1)
        scale_idx = min(scale_idx, self.num_scales - 1)

        # Get slice indices
        start, end = self.slice_indices[layer_idx]

        # Slice tokens and reshape for broadcasting
        cls_token = self.cls_emb[:, task_idx, start:end]  # [1, C]
        sls_token = self.sls_emb[:, scale_idx, start:end]  # [1, C]

        # Expand for batch and reshape to [B, C, 1, 1]
        cls_token = cls_token.unsqueeze(-1).unsqueeze(-1).expand(B, -1, 1, 1)
        sls_token = sls_token.unsqueeze(-1).unsqueeze(-1).expand(B, -1, 1, 1)

        return cls_token, sls_token

    def forward(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
        scale_id: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with layer-wise token injection.

        Args:
            x: Input image [B, 3, H, W]
            task_id: Task identifier [B]
            scale_id: Scale identifier [B]

        Returns:
            features: List of multi-scale features [skip0, skip1, skip2, skip3, bottleneck]
            x_feat: Global pooled features for dynamic head [B, 256, 1, 1]
            cls_token_head: Class token for head [B, 256, 1, 1]
            sls_token_head: Scale token for head [B, 256, 1, 1]
        """
        # Initial conv
        x = self.conv1(x)

        # Layer 0 with token injection (32 channels)
        cls_tok0, sls_tok0 = self._get_token_slice(task_id, scale_id, 0)
        x = self.layer0(x + cls_tok0 + sls_tok0)
        skip0 = x

        # Layer 1 with token injection (32 -> 64 channels)
        cls_tok1, sls_tok1 = self._get_token_slice(task_id, scale_id, 1)
        x = self.layer1(x + cls_tok1 + sls_tok1)
        skip1 = x

        # Layer 2 with token injection (64 -> 128 channels)
        cls_tok2, sls_tok2 = self._get_token_slice(task_id, scale_id, 2)
        x = self.layer2(x + cls_tok2 + sls_tok2)
        skip2 = x

        # Layer 3 with token injection (128 -> 256 channels)
        cls_tok3, sls_tok3 = self._get_token_slice(task_id, scale_id, 3)
        x = self.layer3(x + cls_tok3 + sls_tok3)
        skip3 = x

        # Layer 4 with token injection (256 -> 256 channels)
        cls_tok4, sls_tok4 = self._get_token_slice(task_id, scale_id, 4)
        x = self.layer4(x + cls_tok4 + sls_tok4)

        # Fusion conv
        x = self.fusionConv(x)

        # Global pooling for dynamic head
        x_feat = self.GAP(x)

        # Get tokens for dynamic head (256 channels)
        cls_token_head, sls_token_head = self._get_token_slice(task_id, scale_id, 5)

        return [skip0, skip1, skip2, skip3, x], x_feat, cls_token_head, sls_token_head


class HATsDecoder(nn.Module):
    """
    HATs-style Decoder with skip connections and dynamic head.

    Implements the decoder structure from original HATs with upsampling
    and skip connection fusion.

    Enhanced version: Configurable output channels for dynamic head.
    """

    def __init__(self, head_channels: int = 16):
        super().__init__()

        self.head_channels = head_channels
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder blocks (reverse of encoder)
        self.x8_resb = self._make_layer(256, 128)  # skip3 fusion
        self.x4_resb = self._make_layer(128, 64)   # skip2 fusion
        self.x2_resb = self._make_layer(64, 32)    # skip1 fusion
        self.x1_resb = self._make_layer(32, 32)    # skip0 fusion

        # Pre-classification conv (reduces to head_channels for dynamic head)
        # Increased from 8 to head_channels for more capacity
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, head_channels, kernel_size=1)
        )

    def _make_layer(self, inplanes: int, planes: int) -> nn.Sequential:
        downsample = None
        if inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(min(16, inplanes), inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            )
        return nn.Sequential(NoBottleneck(inplanes, planes, downsample=downsample))

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            features: [skip0, skip1, skip2, skip3, bottleneck]

        Returns:
            head_inputs: Features for dynamic head [B, 8, H, W]
        """
        skip0, skip1, skip2, skip3, x = features

        # x8 -> x4
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        # x4 -> x2
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        # x2 -> x1
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        # x1 -> output
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        # Reduce channels for dynamic head
        head_inputs = self.precls_conv(x)

        return head_inputs


class DynamicHead(nn.Module):
    """
    Dynamic Head that generates segmentation weights based on task/scale.

    Key innovation from HATs: Instead of a fixed segmentation head,
    weights are GENERATED dynamically based on the task and scale tokens.
    This allows the network to adapt its final predictions to different
    structures and magnifications.

    Architecture:
    - Controller generates parameters dynamically
    - These become weights/biases for a 3-layer conv: in->hidden->hidden->2

    Enhanced version: Increased hidden channels from 8 to 16 for more capacity.
    This provides 4x more representational power for distinguishing between
    6 different structure types.
    """

    def __init__(self, in_channels: int = 256, head_channels: int = 16):
        super().__init__()

        self.head_channels = head_channels

        # Dynamic head structure: head_channels input, head_channels hidden, 2 output
        # With head_channels=16: 16*16 + 16*16 + 16*2 = 256 + 256 + 32 = 544 weights
        # Biases: 16 + 16 + 2 = 34
        # Total: 578 parameters (vs 162 with head_channels=8)
        self.weight_nums = [head_channels * head_channels,
                           head_channels * head_channels,
                           head_channels * 2]
        self.bias_nums = [head_channels, head_channels, 2]
        self.total_params = sum(self.weight_nums) + sum(self.bias_nums)

        # Controller: takes conditioned features and outputs head parameters
        # Input: x_feat (256) + cls_token (256) + sls_token (256) = 768
        # But original HATs adds them: 256 channels
        self.controller = nn.Conv2d(in_channels, self.total_params,
                                    kernel_size=1, stride=1, padding=0)

    def parse_dynamic_params(
        self,
        params: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Parse controller output into weights and biases.

        Args:
            params: Controller output [B, total_params]

        Returns:
            weights: List of weight tensors for each layer
            biases: List of bias tensors for each layer
        """
        assert params.dim() == 2
        assert params.size(1) == self.total_params

        num_insts = params.size(0)
        num_layers = len(self.weight_nums)
        channels = self.head_channels

        # Split parameters
        params_splits = list(torch.split_with_sizes(
            params, self.weight_nums + self.bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        # Reshape for grouped convolution
        for l in range(num_layers):
            if l < num_layers - 1:
                # Hidden layers: channels -> channels
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # Output layer: channels -> 2
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(
        self,
        features: torch.Tensor,
        weights: List[torch.Tensor],
        biases: List[torch.Tensor],
        num_insts: int
    ) -> torch.Tensor:
        """
        Apply dynamic head with generated weights.

        Args:
            features: Input features [1, B*8, H, W]
            weights: Generated weights
            biases: Generated biases
            num_insts: Batch size

        Returns:
            logits: Segmentation logits [B, 2, H, W]
        """
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(
        self,
        head_inputs: torch.Tensor,
        x_feat: torch.Tensor,
        cls_token: torch.Tensor,
        sls_token: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate and apply dynamic head.

        Args:
            head_inputs: Decoder output [B, head_channels, H, W]
            x_feat: Global pooled features [B, 256, 1, 1]
            cls_token: Class token [B, 256, 1, 1]
            sls_token: Scale token [B, 256, 1, 1]

        Returns:
            logits: Segmentation logits [B, 2, H, W]
        """
        N, _, H, W = head_inputs.size()

        # Condition features with tokens (ADDITION like original HATs)
        x_cond = x_feat + cls_token + sls_token

        # Generate head parameters
        params = self.controller(x_cond)
        params = params.squeeze(-1).squeeze(-1)  # [B, total_params]

        # Parse into weights and biases
        weights, biases = self.parse_dynamic_params(params)

        # Reshape input for grouped convolution
        head_inputs_grouped = head_inputs.reshape(1, -1, H, W)  # [1, B*head_channels, H, W]

        # Apply dynamic head
        logits = self.heads_forward(head_inputs_grouped, weights, biases, N)
        logits = logits.reshape(N, 2, H, W)

        return logits


class HATsModel(nn.Module):
    """
    Complete HATs Model with layer-wise token injection, transformer, and dynamic head.

    Architecture:
    1. CNN encoder with token injection at every layer (class/scale conditioning)
    2. Transformer at bottleneck for global context
    3. Decoder with skip connections
    4. Dynamic head conditioned on task/scale

    Enhanced version:
    - Increased transformer layers: 2 -> 4 (more global context)
    - Increased dynamic head channels: 8 -> 16 (more task-specific capacity)

    Args:
        num_classes: Number of task classes (6)
        num_scales: Number of scale levels (4)
        use_transformer: Whether to use transformer at bottleneck
        num_transformer_layers: Number of transformer layers (default 4, increased from 2)
        head_channels: Channels for dynamic head (default 16, increased from 8)
        dropout: Dropout rate
    """

    def __init__(self, num_classes: int = 6, num_scales: int = 4,
                 use_transformer: bool = True, num_transformer_layers: int = 4,
                 head_channels: int = 16, dropout: float = 0.1):
        super().__init__()

        self.use_transformer = use_transformer
        self.head_channels = head_channels

        self.encoder = HATsEncoder(
            in_channels=3,
            num_classes=num_classes,
            num_scales=num_scales,
            layers=[1, 2, 2, 2, 2]
        )

        # Transformer at bottleneck for global context
        # Increased from 2 to 4 layers for better global reasoning
        if use_transformer:
            self.bottleneck_transformer = BottleneckTransformer(
                embed_dim=256,
                num_heads=8,
                num_layers=num_transformer_layers,
                dropout=dropout
            )

        self.decoder = HATsDecoder(head_channels=head_channels)
        self.dynamic_head = DynamicHead(in_channels=256, head_channels=head_channels)

    def forward(
        self,
        image: torch.Tensor,
        task_id: torch.Tensor,
        scale_id: torch.Tensor
    ) -> dict:
        """
        Forward pass.

        Args:
            image: Input image [B, 3, H, W]
            task_id: Task identifier [B]
            scale_id: Scale identifier [B]

        Returns:
            Dictionary with 'seg_logits' key
        """
        # Normalize input
        if image.max() > 1.0:
            image = image / 255.0

        # Encode with layer-wise token injection
        features, x_feat, cls_token, sls_token = self.encoder(image, task_id, scale_id)
        # features = [skip0, skip1, skip2, skip3, bottleneck]

        # Apply transformer at bottleneck for global context
        if self.use_transformer:
            # Transform bottleneck features
            bottleneck = features[-1]  # [B, 256, H/16, W/16]
            bottleneck = self.bottleneck_transformer(bottleneck)
            features[-1] = bottleneck

            # Update x_feat after transformer (re-pool)
            x_feat = F.adaptive_avg_pool2d(bottleneck, (1, 1))

        # Decode
        head_inputs = self.decoder(features)

        # Dynamic head for segmentation
        logits = self.dynamic_head(head_inputs, x_feat, cls_token, sls_token)

        # Upsample to match input resolution if needed
        if logits.shape[2:] != image.shape[2:]:
            logits = F.interpolate(logits, size=image.shape[2:],
                                   mode='bilinear', align_corners=False)

        return {
            'seg_logits': logits,
            'boundary_logits': torch.zeros_like(logits[:, :1]),  # Placeholder
            'aux_logits': logits  # For deep supervision compatibility
        }

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_hats_model(
    num_classes: int = 6,
    num_scales: int = 4,
    use_transformer: bool = True,
    num_transformer_layers: int = 4,
    head_channels: int = 16,
    dropout: float = 0.1
) -> HATsModel:
    """
    Build the HATs model with optional transformer.

    Enhanced version with:
    - 4 transformer layers (up from 2) for better global context
    - 16 dynamic head channels (up from 8) for more task-specific capacity

    Args:
        num_classes: Number of task classes (6)
        num_scales: Number of scale levels (4)
        use_transformer: Whether to use transformer at bottleneck
        num_transformer_layers: Number of transformer layers (default 4)
        head_channels: Channels for dynamic head (default 16)
        dropout: Dropout rate
    """
    return HATsModel(
        num_classes=num_classes,
        num_scales=num_scales,
        use_transformer=use_transformer,
        num_transformer_layers=num_transformer_layers,
        head_channels=head_channels,
        dropout=dropout
    )


if __name__ == '__main__':
    # Test
    print("=" * 60)
    print("Testing Enhanced HATsModel with Transformer")
    print("=" * 60)

    model = build_hats_model(
        num_classes=6,
        num_scales=4,
        use_transformer=True,
        num_transformer_layers=4,  # Increased from 2
        head_channels=16  # Increased from 8
    )
    print(f"\nTotal parameters: {model.get_num_parameters():,}")
    print(f"Transformer layers: 4 (was 2)")
    print(f"Dynamic head channels: 16 (was 8)")

    # Test forward
    x = torch.randn(2, 3, 512, 512)
    task_id = torch.tensor([0])
    scale_id = torch.tensor([1])

    print(f"\nInput shape: {x.shape}")
    print(f"Task ID: {task_id.item()}, Scale ID: {scale_id.item()}")

    with torch.no_grad():
        outputs = model(x, task_id, scale_id)

    print(f"\nOutput shapes:")
    print(f"  seg_logits: {outputs['seg_logits'].shape}")
    print(f"  boundary_logits: {outputs['boundary_logits'].shape}")
    print(f"  aux_logits: {outputs['aux_logits'].shape}")

    # Test all task classes
    print("\n" + "=" * 60)
    print("Testing all task classes")
    print("=" * 60)
    class_names = ['DT', 'PT', 'CAP', 'TUFT', 'VES', 'PTC']
    for task_idx in range(6):
        task_id = torch.tensor([task_idx])
        scale_id = torch.tensor([task_idx % 4])
        with torch.no_grad():
            outputs = model(x, task_id, scale_id)
        print(f"Task {task_idx} ({class_names[task_idx]}): output shape = {outputs['seg_logits'].shape}")

    print("\nTest passed!")
