# HybridScaleTransformer: CNN-Transformer Hybrid for Multi-Scale Kidney Pathology Segmentation

A state-of-the-art hybrid CNN-Transformer architecture designed for multi-scale kidney structure segmentation in pathology images.

## Key Features

### Architecture Innovations (Inspired by CVPR/MICCAI Papers)

1. **Scale-Aware Deformable Attention** (inspired by Deformable DETR, ICLR 2021)
   - Adapts receptive field based on object scale
   - Larger sampling offsets for small objects (PTC)
   - Smaller offsets for large structures (TUFT, CAP)

2. **Multi-Scale CNN Encoder** (inspired by FPN, CVPR 2017)
   - Hierarchical feature extraction at 4 scales
   - Scale-conditioned convolutions with adaptive dilation
   - ASPP module for multi-scale context

3. **Class/Scale Token Modulation** (improved from HATs)
   - Learnable embeddings for each class and scale
   - FiLM-style feature modulation
   - Dynamic head parameter generation

4. **Boundary-Aware Segmentation** (inspired by MICCAI papers)
   - Explicit boundary detection module
   - Auxiliary boundary supervision
   - Edge-weighted loss for precise boundaries

### Anti-Overfitting Techniques

1. **Class-Balanced Sampling**
   - Oversamples minority classes (PTC: 96 samples)
   - Effective number weighting (CVPR 2019)

2. **Scale-Aware Loss Weighting**
   - Higher weights for small objects (scale_id=3)
   - Combined class + scale weighting

3. **Strong Augmentation for Rare Classes**
   - Elastic deformation for PTC
   - Aggressive color/noise augmentation

4. **Regularization**
   - Dropout throughout the network
   - Weight decay (AdamW)
   - Learning rate warmup + cosine decay
   - Early stopping

## Dataset Structure

| Task ID | Class | Full Name | Scale ID | Magnification | Train Samples |
|---------|-------|-----------|----------|---------------|---------------|
| 0 | DT | Distal Tubular | 1 | 10x | 285 |
| 1 | PT | Proximal Tubular | 1 | 10x | 285 |
| 2 | CAP | Glomerular Unit | 0 | 5x | 369 |
| 3 | TUFT | Glomerular Tuft | 0 | 5x | 388 |
| 4 | VES | Arteries | 1 | 10x | 182 |
| 5 | PTC | Peritubular Capillaries | 3 | 40x | 96 |

### Object Characteristics
- **Large Objects** (TUFT, CAP): Single blob-like structures
- **Medium Objects** (DT, PT, VES): Multiple scattered structures
- **Small Objects** (PTC): Many tiny scattered structures requiring 40x magnification

## Installation

```bash
# Clone the repository
cd HATs-main/hybrid_cnn_transformer

# Install dependencies
pip install torch torchvision
pip install imgaug
pip install pandas numpy scipy opencv-python pillow
```

## Usage

### Training

```bash
python train.py \
    --train_csv ../train_list.csv \
    --val_csv ../val_list.csv \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --class_balanced \
    --use_transformer
```

### Inference

```python
from models import build_hybrid_model
from utils import load_model, predict_single, sliding_window_inference

# Load model
model = load_model('checkpoints/best_model.pth')

# Single image prediction
result = predict_single(model, image, task_id=5, scale_id=3)
mask = result['mask']

# Large image sliding window inference
result = sliding_window_inference(
    model, large_image,
    task_id=5, scale_id=3,
    window_size=(512, 512),
    stride=(256, 256)
)
```

### Evaluation

```python
from utils import evaluate_model
from datasets import PathologyValDataset, collate_fn
from torch.utils.data import DataLoader

val_dataset = PathologyValDataset('val_list.csv')
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

metrics = evaluate_model(model, val_loader)
print(f"Overall Dice: {metrics['dice']:.4f}")
print(f"PTC Dice: {metrics['PTC_dice']:.4f}")
```

## Model Architecture

```
HybridScaleTransformer
├── MultiScaleCNNEncoder
│   ├── Stem (3 → 64, 1/4 resolution)
│   ├── Stage1 (64 → 64, 1/4)
│   ├── Stage2 (64 → 128, 1/8)
│   ├── Stage3 (128 → 256, 1/16)
│   ├── Stage4 (256 → 256, 1/32)
│   ├── ScaleConditionedConv
│   ├── ASPP
│   └── Class/Scale Embeddings
│
├── ScaleAwareTransformerEncoder
│   ├── Stage1: EfficientAttention (high-res)
│   ├── Stage2: EfficientAttention
│   ├── Stage3: DeformableAttention (low-res)
│   └── Stage4: DeformableAttention
│
└── MultiScaleDecoder
    ├── FeaturePyramidFusion (FPN)
    ├── Progressive Upsampling
    ├── BoundaryAwareHead
    └── DynamicHead (class/scale conditioned)
```

## Loss Function

Combined loss with multiple components:

```
L_total = w_dice * L_dice + w_focal * L_focal + w_boundary * L_boundary + w_aux * L_aux
```

- **Dice Loss**: Handles class imbalance
- **Focal Loss**: Hard example mining (γ=2.0)
- **Boundary Loss**: Precise edge segmentation
- **Auxiliary Loss**: Deep supervision

With class-balanced weighting based on effective number of samples.

## Results Comparison

| Model | DT | PT | CAP | TUFT | VES | PTC | Mean |
|-------|-----|-----|-----|------|-----|-----|------|
| Original UNet2D | - | - | - | - | - | - | - |
| EfficientSAM HATs | - | - | - | - | - | - | - |
| **HybridScaleTransformer** | - | - | - | - | - | - | - |

*(Fill in with actual results after training)*

## Key Improvements Over Original HATs

1. **Better Scale Token Usage**: Original code concatenates tokens but doesn't fully leverage them. Our model uses:
   - FiLM-style modulation
   - Scale-aware deformable attention
   - Dynamic head conditioning

2. **Explicit Multi-Scale Handling**:
   - Different dilation rates per scale
   - Adaptive receptive field via deformable attention

3. **Class Imbalance Handling**:
   - Class-balanced sampling
   - Scale-aware loss weighting
   - Stronger augmentation for PTC

4. **Boundary Awareness**:
   - Explicit boundary detection module
   - Boundary loss for small structure segmentation

## Directory Structure

```
hybrid_cnn_transformer/
├── models/
│   ├── __init__.py
│   ├── hybrid_scale_transformer.py  # Main model
│   ├── cnn_encoder.py               # CNN backbone
│   ├── transformer_blocks.py        # Transformer modules
│   └── decoder.py                   # Decoder with boundary head
├── losses/
│   ├── __init__.py
│   └── combined_loss.py             # Multi-component loss
├── datasets/
│   ├── __init__.py
│   └── pathology_dataset.py         # Data loading
├── utils/
│   ├── __init__.py
│   └── inference.py                 # Inference utilities
├── train.py                         # Training script
└── README.md
```

## References

1. TransUNet (CVPR 2021): Transformers for medical image segmentation
2. SegFormer (NeurIPS 2021): Efficient hierarchical transformer
3. Deformable DETR (ICLR 2021): Deformable attention mechanism
4. HATs (CVPR 2024): Hierarchical adaptive taxonomy segmentation
5. Class-Balanced Loss (CVPR 2019): Effective number for long-tailed recognition
6. Focal Loss (ICCV 2017): Dense object detection

## License

This project is licensed under the MIT License.
