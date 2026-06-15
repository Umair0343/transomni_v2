# HybridScaleTransformer: Complete Model Explanation

This document provides a detailed step-by-step explanation of how the HybridScaleTransformer processes an input image, with special focus on how the **learnable class and scale tokens** work.

---

## Table of Contents
1. [Model Overview](#1-model-overview)
2. [Input Processing](#2-input-processing)
3. [CNN Encoder Stage](#3-cnn-encoder-stage)
4. [Learnable Tokens Explained](#4-learnable-tokens-explained)
5. [Transformer Encoder Stage](#5-transformer-encoder-stage)
6. [Decoder Stage](#6-decoder-stage)
7. [Dynamic Head](#7-dynamic-head)
8. [Complete Forward Pass Example](#8-complete-forward-pass-example)

---

## 1. Model Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HybridScaleTransformer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT                                                                      │
│   ├── image: [B, 3, 512, 512]     (RGB pathology patch)                     │
│   ├── task_id: [B]                 (0=DT, 1=PT, 2=CAP, 3=TUFT, 4=VES, 5=PTC)│
│   └── scale_id: [B]                (0=5x, 1=10x, 2=20x, 3=40x)              │
│                                                                              │
│   ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐                 │
│   │ CNN Encoder │───▶│ Transformer Enc │───▶│   Decoder   │───▶ OUTPUT     │
│   └─────────────┘    └─────────────────┘    └─────────────┘                 │
│         │                    │                     │                         │
│         └────────── Learnable Tokens ──────────────┘                         │
│                   (cls_emb + sls_emb)                                        │
│                                                                              │
│   OUTPUT                                                                     │
│   ├── seg_logits: [B, 2, 512, 512]     (segmentation prediction)            │
│   ├── boundary_logits: [B, 1, 512, 512] (boundary prediction)               │
│   └── aux_logits: [B, 2, 512, 512]      (auxiliary for deep supervision)    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Input Processing

### Step 2.1: Image Normalization
```python
# Input image: [B, 3, 512, 512] with values in [0, 255]
# Normalize to [0, 1]
if image.max() > 1.0:
    image = image / 255.0

# Result: [B, 3, 512, 512] with values in [0, 1]
```

### Step 2.2: Task and Scale ID
```
Example:
- task_id = 5 (PTC - Peritubular Capillaries)
- scale_id = 3 (40x magnification)

These IDs are used to SELECT specific learnable tokens from embedding tables.
```

---

## 3. CNN Encoder Stage

The CNN encoder extracts hierarchical features at multiple scales.

### Architecture Flow:
```
Input Image [B, 3, 512, 512]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STEM: Conv2d(3→64, 7x7, stride=2) + MaxPool(3x3, stride=2) │
│  Output: [B, 64, 128, 128]  (1/4 resolution)                │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: 2x ResidualBlock(64→64)                           │
│  Output: C1 = [B, 64, 128, 128]  (1/4 resolution)           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: 2x ResidualBlock(64→128, stride=2)                │
│  Output: C2 = [B, 128, 64, 64]   (1/8 resolution)           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: 2x ResidualBlock(128→256, stride=2)               │
│  Output: C3 = [B, 256, 32, 32]   (1/16 resolution)          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: 2x ResidualBlock(256→512, stride=2)               │
│  Output: C4 = [B, 512, 16, 16]   (1/32 resolution)          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  SCALE-CONDITIONED CONV + ASPP                               │
│  - Uses scale_id to adapt dilation rates                     │
│  - ASPP provides multi-scale context                         │
│  Output: C4' = [B, 256, 16, 16]                              │
└─────────────────────────────────────────────────────────────┘
```

### Scale-Conditioned Convolution Detail:
```python
# Different dilation rates for different scales
dilations = {
    scale_id=0 (5x):  dilation=1  # Small dilation for LARGE objects (TUFT, CAP)
    scale_id=1 (10x): dilation=2  # Medium dilation for MEDIUM objects (DT, PT, VES)
    scale_id=2 (20x): dilation=3  # Larger dilation
    scale_id=3 (40x): dilation=6  # LARGEST dilation for SMALL objects (PTC)
}

# Why? Larger dilation = larger receptive field = captures more context
# Small objects (PTC) need MORE context to understand their surroundings
```

---

## 4. Learnable Tokens Explained

This is the KEY innovation that makes the model adapt to different classes and scales.

### 4.1 What are Learnable Tokens?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEARNABLE TOKEN TABLES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CLASS EMBEDDINGS (cls_emb):                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Shape: [1, 6, 512]  (6 classes, 512-dim embedding each)            │    │
│  │                                                                      │    │
│  │  Index 0: DT token    ████████████████████ (512 learnable params)   │    │
│  │  Index 1: PT token    ████████████████████ (512 learnable params)   │    │
│  │  Index 2: CAP token   ████████████████████ (512 learnable params)   │    │
│  │  Index 3: TUFT token  ████████████████████ (512 learnable params)   │    │
│  │  Index 4: VES token   ████████████████████ (512 learnable params)   │    │
│  │  Index 5: PTC token   ████████████████████ (512 learnable params)   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  SCALE EMBEDDINGS (sls_emb):                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Shape: [1, 4, 512]  (4 scales, 512-dim embedding each)             │    │
│  │                                                                      │    │
│  │  Index 0: 5x token    ████████████████████ (512 learnable params)   │    │
│  │  Index 1: 10x token   ████████████████████ (512 learnable params)   │    │
│  │  Index 2: 20x token   ████████████████████ (512 learnable params)   │    │
│  │  Index 3: 40x token   ████████████████████ (512 learnable params)   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  TOTAL LEARNABLE PARAMETERS: 6*512 + 4*512 = 5,120 parameters               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 How Tokens are Selected

```python
# Given: task_id = 5 (PTC), scale_id = 3 (40x)

# Step 1: Select class token
cls_token = cls_emb[:, task_id, :]  # cls_emb[:, 5, :] → [1, 512]
# This is the LEARNED representation for "PTC structures"

# Step 2: Select scale token
sls_token = sls_emb[:, scale_id, :]  # sls_emb[:, 3, :] → [1, 512]
# This is the LEARNED representation for "40x magnification"

# These tokens ENCODE what the model has LEARNED about:
# - How PTC structures look (shape, texture, context)
# - How 40x images differ from 5x or 10x images
```

### 4.3 How Tokens Modulate Features

The tokens influence the network in THREE ways:

#### Method 1: FiLM-Style Modulation (Feature-wise Linear Modulation)
```python
# Combine class and scale tokens
combined_token = cls_token + sls_token  # [1, 512]

# Generate modulation parameters (gamma and beta)
modulation = MLP(combined_token)  # [1, 1024]
gamma, beta = modulation.chunk(2)  # Each [1, 512]

# Apply to features (like batch normalization but learned per-class)
features_modulated = gamma * features + beta

# Effect: The network learns different "adjustments" for each class
# - PTC features might be ENHANCED differently than TUFT features
# - 40x features might be SCALED differently than 5x features
```

```
VISUAL EXAMPLE OF FiLM MODULATION:

Original Feature Map (C4):        After FiLM Modulation:
┌─────────────────────┐           ┌─────────────────────┐
│  0.2  0.5  0.1  0.3 │           │  0.4  1.0  0.2  0.6 │  ← gamma=2.0
│  0.8  0.4  0.6  0.2 │  ────▶    │  1.6  0.8  1.2  0.4 │  (amplify)
│  0.1  0.7  0.3  0.5 │  gamma,   │  0.2  1.4  0.6  1.0 │
│  0.4  0.2  0.8  0.1 │  beta     │  0.8  0.4  1.6  0.2 │
└─────────────────────┘           └─────────────────────┘

For PTC (small objects): gamma might AMPLIFY edge features
For TUFT (large objects): gamma might SMOOTH blob features
```

#### Method 2: Attention Token Injection
```python
# In Transformer blocks, tokens are concatenated with image features

# Image features as sequence: [B, H*W, C] = [B, 1024, 256] for 32x32 feature map
# Class token: [1, 1, 256]
# Scale token: [1, 1, 256]

# Concatenate for attention:
sequence = torch.cat([cls_token, sls_token, image_features], dim=1)
# Shape: [B, 2 + 1024, 256] = [B, 1026, 256]

# Self-attention computes relationships:
attention_output = MultiHeadAttention(sequence)

# The image features can now ATTEND to the class/scale tokens
# Learning: "Given I'm looking at PTC at 40x, what should I focus on?"
```

```
ATTENTION VISUALIZATION:

Query (image patch) attends to:
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────┐     ┌─────────┐     ┌─────────────────────┐   │
│  │ CLS_TOK │     │ SLS_TOK │     │   Image Patches     │   │
│  │  (PTC)  │     │  (40x)  │     │   [1024 tokens]     │   │
│  └────┬────┘     └────┬────┘     └──────────┬──────────┘   │
│       │               │                      │              │
│       ▼               ▼                      ▼              │
│   Attention      Attention              Self-Attention      │
│   weight=0.3     weight=0.2             weight=0.5          │
│                                                             │
│  The patch learns: "I should look for small scattered       │
│  structures (from PTC token) at high magnification          │
│  (from 40x token)"                                          │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

#### Method 3: Dynamic Head Conditioning
```python
# Tokens condition the final segmentation head

# Global feature + tokens
x_cond = torch.cat([global_avg_pool(features), cls_token, sls_token], dim=1)
# Shape: [B, 256 + 512 + 512] = [B, 1280]

# Generate dynamic convolution parameters
params = controller_MLP(x_cond)  # [B, 162]
# 162 = 64 + 64 + 16 + 8 + 8 + 2 (weights + biases for 3 conv layers)

# These parameters CREATE a custom segmentation head for THIS class/scale
weights, biases = parse_params(params)

# Apply dynamic convolution
output = DynamicConv(features, weights, biases)
```

```
DYNAMIC HEAD VISUALIZATION:

For TUFT (large blob):              For PTC (small scattered):
┌─────────────────────┐             ┌─────────────────────┐
│ Generated weights   │             │ Generated weights   │
│ favor SMOOTH edges  │             │ favor SHARP edges   │
│ and LARGE regions   │             │ and SMALL regions   │
└─────────────────────┘             └─────────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────────┐             ┌─────────────────────┐
│   Prediction:       │             │   Prediction:       │
│   ┌───────────┐     │             │   · · · · · · ·     │
│   │           │     │             │   · · · · · · ·     │
│   │   BLOB    │     │             │   · · · · · · ·     │
│   │           │     │             │   · · · · · · ·     │
│   └───────────┘     │             │   (many small dots) │
└─────────────────────┘             └─────────────────────┘
```

### 4.4 What Do Tokens Learn?

During training, the tokens learn to encode:

```
CLASS TOKENS LEARN:                    SCALE TOKENS LEARN:
┌────────────────────────────┐        ┌────────────────────────────┐
│ DT token:                  │        │ 5x token:                  │
│ - Tubular shape patterns   │        │ - Low detail features      │
│ - Medium-sized scattered   │        │ - Large context needed     │
│ - ~29% coverage expected   │        │ - Smooth boundaries        │
├────────────────────────────┤        ├────────────────────────────┤
│ PTC token:                 │        │ 40x token:                 │
│ - Tiny scattered patterns  │        │ - High detail features     │
│ - Very small objects       │        │ - Local context important  │
│ - ~5% coverage expected    │        │ - Sharp boundaries         │
├────────────────────────────┤        ├────────────────────────────┤
│ TUFT token:                │        │ 10x token:                 │
│ - Large blob patterns      │        │ - Medium detail            │
│ - Single object per image  │        │ - Balanced context         │
│ - ~9% coverage expected    │        │ - Moderate boundaries      │
└────────────────────────────┘        └────────────────────────────┘
```

---

## 5. Transformer Encoder Stage

### Architecture:
```
Input: Multi-scale features [C1, C2, C3, C4] + cls_token + sls_token

┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Process C1 [B, 64, 128, 128]                                      │
│  ├── Add 2D Positional Encoding                                             │
│  ├── Flatten to sequence: [B, 16384, 64]                                    │
│  ├── Inject tokens: [B, 16386, 64]                                          │
│  ├── EfficientAttention (spatial reduction for efficiency)                  │
│  └── Output: P1 [B, 64, 128, 128]                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  STAGE 2: Process C2 [B, 128, 64, 64]                                       │
│  ├── Similar processing with EfficientAttention                             │
│  └── Output: P2 [B, 128, 64, 64]                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  STAGE 3: Process C3 [B, 256, 32, 32]                                       │
│  ├── DeformableAttention (scale-aware sampling)                             │
│  ├── Sampling offsets scaled by scale_id                                    │
│  └── Output: P3 [B, 256, 32, 32]                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  STAGE 4: Process C4 [B, 256, 16, 16]                                       │
│  ├── DeformableAttention with largest receptive field                       │
│  └── Output: P4 [B, 256, 16, 16]                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Deformable Attention Detail:
```
Standard Attention:              Deformable Attention:
┌─────────────────────┐         ┌─────────────────────┐
│ ● ● ● ● ● ● ● ● ●  │         │ ●       ●       ●   │
│ ● ● ● ● ● ● ● ● ●  │         │                     │
│ ● ● ● ● ● ● ● ● ●  │         │     ●       ●      │
│ ● ● ● ● Q ● ● ● ●  │         │         Q          │
│ ● ● ● ● ● ● ● ● ●  │         │     ●       ●      │
│ ● ● ● ● ● ● ● ● ●  │         │                     │
│ ● ● ● ● ● ● ● ● ●  │         │ ●       ●       ●   │
└─────────────────────┘         └─────────────────────┘
Attends to ALL positions        Attends to LEARNED positions
(O(N²) complexity)              (O(N×K) complexity, K=4 points)

For PTC (scale_id=3):           For TUFT (scale_id=0):
┌─────────────────────┐         ┌─────────────────────┐
│ ●               ●   │         │     ● ● ●           │
│                     │         │     ● Q ●           │
│         Q           │         │     ● ● ●           │
│                     │         │                     │
│ ●               ●   │         │                     │
└─────────────────────┘         └─────────────────────┘
LARGE offsets (more context)    SMALL offsets (local detail)
```

---

## 6. Decoder Stage

### Feature Pyramid Network (FPN) Fusion:
```
        P4 [256, 16, 16]
              │
              ▼ Upsample 2x
        ┌─────────────┐
        │     +       │◄── P3 [256, 32, 32] (lateral connection)
        └─────────────┘
              │
              ▼ Upsample 2x
        ┌─────────────┐
        │     +       │◄── P2 [128→256, 64, 64]
        └─────────────┘
              │
              ▼ Upsample 2x
        ┌─────────────┐
        │     +       │◄── P1 [64→256, 128, 128]
        └─────────────┘
              │
              ▼ Upsample 4x
        Output [256, 512, 512]
```

### Boundary-Aware Head:
```
Features [B, 256, 512, 512]
         │
         ├────────────────────────┐
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│ Boundary Branch │      │   Seg Branch    │
│                 │      │                 │
│ Laplacian Conv  │      │ Features + Bnd  │
│ Gradient Conv   │      │ Conv layers     │
│                 │      │                 │
│ Output: [B,1,H,W]      │ Output: [B,2,H,W]
│ (boundary map)  │      │ (seg logits)    │
└─────────────────┘      └─────────────────┘
```

---

## 7. Dynamic Head

The dynamic head generates CLASS-SPECIFIC and SCALE-SPECIFIC convolution weights:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DYNAMIC HEAD GENERATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Global features + cls_token + sls_token                             │
│         [B, 256] + [1, 512] + [1, 512] = [B, 1280]                          │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                    ┌─────────────────┐                                       │
│                    │   Controller    │                                       │
│                    │   MLP Network   │                                       │
│                    │                 │                                       │
│                    │ Linear(1280→512)│                                       │
│                    │ ReLU            │                                       │
│                    │ Linear(512→162) │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│                             ▼                                                │
│                    162 parameters                                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PARAMETER BREAKDOWN                               │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │  Layer 1: 8×8 weights + 8 biases  = 64 + 8  = 72 params             │    │
│  │  Layer 2: 8×8 weights + 8 biases  = 64 + 8  = 72 params             │    │
│  │  Layer 3: 8×2 weights + 2 biases  = 16 + 2  = 18 params             │    │
│  │  ─────────────────────────────────────────────────                  │    │
│  │  TOTAL:                                        162 params            │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DYNAMIC CONVOLUTION                               │    │
│  │                                                                      │    │
│  │  Input features: [B, 8, H, W]                                       │    │
│  │                                                                      │    │
│  │  Layer 1: Conv(8→8, 1×1) + ReLU    weights from params[0:64]        │    │
│  │  Layer 2: Conv(8→8, 1×1) + ReLU    weights from params[72:136]      │    │
│  │  Layer 3: Conv(8→2, 1×1)           weights from params[144:162]     │    │
│  │                                                                      │    │
│  │  Output: [B, 2, H, W] (foreground/background logits)                │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Dynamic Head is Powerful:

```
STATIC HEAD (traditional):
- Same convolution weights for ALL classes
- Model must learn "average" filters

DYNAMIC HEAD (our approach):
- DIFFERENT convolution weights for EACH class/scale combination
- PTC gets filters tuned for small objects
- TUFT gets filters tuned for large blobs

Example learned filters:

For PTC (small scattered):        For TUFT (large blob):
┌─────────────┐                   ┌─────────────┐
│ -1  -1  -1  │                   │  0   0   0  │
│ -1   8  -1  │  Edge detector    │  0   1   0  │  Gaussian blur
│ -1  -1  -1  │                   │  0   0   0  │
└─────────────┘                   └─────────────┘
(enhances small structures)       (smooths large regions)
```

---

## 8. Complete Forward Pass Example

Let's trace through a complete example:

```python
# Input
image = torch.randn(1, 3, 512, 512)  # One RGB image
task_id = torch.tensor([5])           # PTC class
scale_id = torch.tensor([3])          # 40x magnification
```

### Step-by-Step:

```
STEP 1: CNN ENCODER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  [1, 3, 512, 512]
        │
        ▼
Stem:   [1, 64, 128, 128]   # 1/4 resolution
        │
        ▼
Stage1: C1 = [1, 64, 128, 128]    ─────────────────────────────────┐
        │                                                           │
        ▼                                                           │
Stage2: C2 = [1, 128, 64, 64]     ───────────────────────┐         │
        │                                                 │         │
        ▼                                                 │         │
Stage3: C3 = [1, 256, 32, 32]     ─────────────┐         │         │
        │                                       │         │         │
        ▼                                       │         │         │
Stage4: C4 = [1, 512, 16, 16]                   │         │         │
        │                                       │         │         │
        ▼                                       │         │         │
Scale-Cond Conv (dilation=6 for 40x)           │         │         │
        │                                       │         │         │
        ▼                                       │         │         │
ASPP:   C4' = [1, 256, 16, 16]                 │         │         │
                                               │         │         │
                                               │         │         │
STEP 2: TOKEN SELECTION                        │         │         │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━│━━━━━━━━━│━━━━━━━━
cls_emb table: [1, 6, 512]                     │         │         │
task_id = 5 (PTC)                              │         │         │
cls_token = cls_emb[:, 5, :] = [1, 512]        │         │         │
                                               │         │         │
sls_emb table: [1, 4, 512]                     │         │         │
scale_id = 3 (40x)                             │         │         │
sls_token = sls_emb[:, 3, :] = [1, 512]        │         │         │
                                               │         │         │
                                               │         │         │
STEP 3: FiLM MODULATION                        │         │         │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━│━━━━━━━━━│━━━━━━━━
combined = cls_token + sls_token = [1, 512]    │         │         │
modulation = MLP(combined) = [1, 1024]         │         │         │
gamma, beta = split(modulation)                │         │         │
                                               │         │         │
C4_modulated = gamma * C4' + beta              │         │         │
             = [1, 256, 16, 16]                │         │         │
                                               │         │         │
                                               │         │         │
STEP 4: TRANSFORMER ENCODER                    │         │         │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━│━━━━━━━━━│━━━━━━━━
                                               │         │         │
P4 = TransformerStage4(C4_mod, cls, sls)  ◄────┘         │         │
   = [1, 256, 16, 16]                                    │         │
                                                         │         │
P3 = TransformerStage3(C3, cls, sls)  ◄──────────────────┘         │
   = [1, 256, 32, 32]                                              │
                                                                   │
P2 = TransformerStage2(C2, cls, sls)  ◄────────────────────────────│
   = [1, 128, 64, 64]                                              │
                                                                   │
P1 = TransformerStage1(C1, cls, sls)  ◄────────────────────────────┘
   = [1, 64, 128, 128]


STEP 5: FPN DECODER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P4 [256, 16, 16]
    │
    ▼ upsample + lateral
F3 = Up(P4) + Lateral(P3) = [256, 32, 32]
    │
    ▼ upsample + lateral
F2 = Up(F3) + Lateral(P2) = [256, 64, 64]
    │
    ▼ upsample + lateral
F1 = Up(F2) + Lateral(P1) = [256, 128, 128]
    │
    ▼ upsample 4x
F0 = [256, 512, 512]
    │
    ▼ pre-head conv
Features = [8, 512, 512]


STEP 6: DYNAMIC HEAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Global feature = GAP(P4) = [1, 256]
Conditioning = [global, cls_token, sls_token] = [1, 1280]
                │
                ▼
        Controller MLP
                │
                ▼
        params = [1, 162]
                │
                ▼
        Parse into weights/biases
                │
                ▼
        DynamicConv(Features, weights, biases)
                │
                ▼
        seg_logits = [1, 2, 512, 512]


STEP 7: BOUNDARY HEAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
F0 [256, 512, 512]
    │
    ▼
Boundary Detector (Laplacian + Sobel)
    │
    ▼
boundary_logits = [1, 1, 512, 512]


FINAL OUTPUT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
    'seg_logits':      [1, 2, 512, 512],   # Main segmentation
    'boundary_logits': [1, 1, 512, 512],   # Boundary prediction
    'aux_logits':      [1, 2, 512, 512],   # Deep supervision
}
```

---

## Summary: Why Learnable Tokens Matter

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LEARNABLE TOKENS: KEY BENEFITS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. TASK-SPECIFIC ADAPTATION                                                 │
│     - Each class has its own learned "encoding"                             │
│     - Network behavior changes based on target structure                    │
│     - PTC processing ≠ TUFT processing                                      │
│                                                                              │
│  2. SCALE-AWARE PROCESSING                                                   │
│     - Different magnifications need different strategies                    │
│     - 40x (small objects) → larger context, sharper edges                   │
│     - 5x (large objects) → local detail, smooth regions                     │
│                                                                              │
│  3. EFFICIENT PARAMETER SHARING                                              │
│     - Single network for ALL classes (not 6 separate networks)              │
│     - Only 5,120 extra parameters for 6 classes × 4 scales                  │
│     - Much more efficient than class-specific models                        │
│                                                                              │
│  4. IMPLICIT KNOWLEDGE TRANSFER                                              │
│     - Learning on DT helps with PT (similar tubular structures)            │
│     - Learning on CAP helps with TUFT (similar glomerular shapes)          │
│     - Tokens capture shared knowledge while allowing specialization        │
│                                                                              │
│  5. DYNAMIC HEAD CUSTOMIZATION                                               │
│     - Final segmentation layer is GENERATED per class/scale                │
│     - Allows fine-grained adaptation without separate models               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Reference

Key files for understanding tokens:
- Token definition: `models/cnn_encoder.py:130-131` (cls_emb, sls_emb)
- Token selection: `models/cnn_encoder.py:209-216`
- FiLM modulation: `models/cnn_encoder.py:224-230`
- Transformer injection: `models/transformer_blocks.py:260-280`
- Dynamic head: `models/decoder.py:220-260`
