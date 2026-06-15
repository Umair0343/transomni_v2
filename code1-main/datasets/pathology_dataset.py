"""
Pathology Dataset for Multi-Scale Kidney Structure Segmentation

Key Features:
1. Scale-Aware Augmentation:
   - Large objects (TUFT, CAP): Standard augmentation
   - Small objects (PTC): Stronger augmentation to increase samples

2. Class-Balanced Sampling:
   - Oversamples minority classes (PTC, VES)
   - Prevents model from ignoring rare structures

3. Smart Patch Extraction:
   - Ensures patches contain meaningful content
   - Avoids patches with >80% foreground (likely annotation errors)
   - NEW: Avoids patches with <1% foreground for single-object classes
   - NEW: Object-aware cropping for CAP/TUFT classes

4. Multi-Stain Handling:
   - Stain normalization for consistency
   - Color augmentation for stain invariance

Dataset Structure:
- task_id: 0=DT, 1=PT, 2=CAP, 3=TUFT, 4=VES, 5=PTC
- scale_id: 0=5x, 1=10x, 2=20x, 3=40x
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Callable
import random
import scipy.ndimage


# Augmentation transforms
try:
    import imgaug.augmenters as iaa
    HAS_IMGAUG = True
except ImportError:
    HAS_IMGAUG = False
    print("Warning: imgaug not installed. Using basic augmentations.")


def get_train_transforms(scale_id: int = 1, strong: bool = False):
    """
    Get training augmentation transforms.

    Args:
        scale_id: Scale identifier (0=5x, 1=10x, 3=40x)
        strong: Whether to use strong augmentation (for minority classes)

    Returns:
        Augmentation pipeline
    """
    if not HAS_IMGAUG:
        return None

    # Base augmentation (geometric) - moderate settings to prevent overfitting
    geometric_aug = iaa.Sequential([
        iaa.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Reduced
            rotate=(-180, 180),
            shear=(-10, 10),  # Reduced from 16
            scale=(0.9, 1.1) if not strong else (0.85, 1.15)  # Much tighter scale range
        ),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])

    # Color augmentation (intensity)
    color_aug = iaa.Sequential([
        iaa.GammaContrast((0.5, 2.0)),
        iaa.Add((-0.15, 0.15), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.3)
    ])

    # Strong augmentation for minority classes
    if strong:
        noise_aug = iaa.Sequential([
            iaa.CoarseDropout((0.0, 0.1), size_percent=(0.02, 0.1)),
            iaa.GaussianBlur(sigma=(0, 1.5)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.15)),
            iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=50, sigma=5))
        ])
    else:
        noise_aug = iaa.Sequential([
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.05)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.1))
        ])

    return {
        'geometric': geometric_aug,
        'color': color_aug,
        'noise': noise_aug
    }


def get_val_transforms():
    """Get validation transforms (no augmentation)"""
    return None


class PathologyDataset(Dataset):
    """
    Training dataset for multi-scale pathology segmentation.

    Features:
    - CSV-based data loading
    - Scale-aware augmentation
    - Smart patch cropping with class-specific strategies
    - Edge weight computation for boundary loss
    """

    # Class information with cropping strategy
    # UPDATED: DT and PT both use sparse_aware (70% random, 20% FG, 10% boundary)
    #
    # KEY INSIGHT from pixel analysis:
    # - DT: SPARSE (~5-10% coverage) - tiny scattered blobs
    # - PT: MODERATE (~30-40% coverage) - larger connected blobs
    # - CAP/TUFT: DENSE single object - object_aware works well
    #
    CLASS_INFO = {
        0: {'name': 'DT', 'scale': 1, 'samples': 285, 'aug_strength': 'strong',
            'crop_strategy': 'dt_aware', 'min_fg': 0.01, 'max_fg': 0.70},  # 30% centered, 30% boundary, 40% random
        1: {'name': 'PT', 'scale': 1, 'samples': 285, 'aug_strength': 'medium',
            'crop_strategy': 'sparse_aware', 'min_fg': 0.01, 'max_fg': 0.70},  # sparse_aware (70% random)
        2: {'name': 'CAP', 'scale': 0, 'samples': 369, 'aug_strength': 'normal',  # FIXED! val > train
            'crop_strategy': 'glomerular_aware', 'min_fg': 0.02, 'max_fg': 0.95},  # 25% centered, 35% boundary, 40% random
        3: {'name': 'TUFT', 'scale': 0, 'samples': 388, 'aug_strength': 'normal', # FIXED! val > train
            'crop_strategy': 'object_aware', 'min_fg': 0.02, 'max_fg': 0.95},
        4: {'name': 'VES', 'scale': 1, 'samples': 182, 'aug_strength': 'medium',
            'crop_strategy': 'vessel_aware', 'min_fg': 0.01, 'max_fg': 0.80},  # 30% FG, 30% boundary, 40% random
        5: {'name': 'PTC', 'scale': 3, 'samples': 96, 'aug_strength': 'strong',
            'crop_strategy': 'ptc_aware', 'min_fg': 0.001, 'max_fg': 0.80}  # 40% FG, 20% boundary, 40% random (better val match)
    }

    def __init__(
        self,
        csv_path: str,
        crop_size: Tuple[int, int] = (512, 512),
        augment: bool = True,
        edge_weight: bool = True,
        max_iters: Optional[int] = None,
        exclude_classes: Optional[List[int]] = None
    ):
        """
        Args:
            csv_path: Path to CSV file with image paths
            crop_size: Size of cropped patches (H, W)
            augment: Whether to apply augmentation
            edge_weight: Whether to compute edge weights
            max_iters: Maximum iterations per epoch (for oversampling)
            exclude_classes: List of task_ids to exclude from training (e.g., [4, 5] for VES, PTC)
        """
        super().__init__()

        self.csv_path = csv_path
        self.crop_h, self.crop_w = crop_size
        self.augment = augment
        self.edge_weight = edge_weight
        self.max_iters = max_iters
        self.exclude_classes = exclude_classes or []

        # Load CSV - auto-detect separator
        self.df = self._load_csv(csv_path)

        # Filter out excluded classes
        if self.exclude_classes:
            original_len = len(self.df)
            self.df = self.df[~self.df['task_id'].isin(self.exclude_classes)].reset_index(drop=True)
            excluded_names = [self.CLASS_INFO.get(c, {}).get('name', f'Class{c}') for c in self.exclude_classes]
            print(f"Excluded classes {excluded_names}: {original_len} -> {len(self.df)} samples")

        print(f"Loaded {len(self.df)} samples from {csv_path}")

        # Analyze class distribution
        self._analyze_distribution()

        # Initialize augmentation transforms
        self.transforms = {
            'normal': get_train_transforms(scale_id=1, strong=False) if augment else None,
            'medium': get_train_transforms(scale_id=1, strong=False) if augment else None,
            'strong': get_train_transforms(scale_id=3, strong=True) if augment else None
        }

        # Crop utilities
        if HAS_IMGAUG:
            self.crop_512 = iaa.CropToFixedSize(width=512, height=512, position='uniform')
            self.pad_512 = iaa.PadToFixedSize(width=512, height=512, position='uniform')

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with auto-detection of separator and column names"""
        # Try tab-separated first, then comma
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                # Check if we have the required columns
                required_cols = ['image_path', 'label_path', 'task_id', 'scale_id']
                if all(col in df.columns for col in required_cols):
                    sep_name = 'tab' if sep == '\t' else 'comma'
                    print("CSV loaded with separator:", sep_name)
                    print("Columns found:", list(df.columns))
                    return df
            except Exception as e:
                continue

        # If standard loading failed, try to infer from file
        df = pd.read_csv(csv_path, sep=None, engine='python')  # Auto-detect
        print(f"CSV auto-detected. Columns: {list(df.columns)}")

        # Handle different column naming conventions
        column_mapping = {
            'img_path': 'image_path',
            'image': 'image_path',
            'mask_path': 'label_path',
            'mask': 'label_path',
            'label': 'label_path',
            'class_id': 'task_id',
            'class': 'task_id',
            'task': 'task_id',
            'scale': 'scale_id',
            'mag': 'scale_id',
            'magnification': 'scale_id',
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
                print(f"Renamed column: {old_name} -> {new_name}")

        # Verify required columns exist
        required_cols = ['image_path', 'label_path', 'task_id', 'scale_id']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}\n"
                           f"Available columns: {list(df.columns)}\n"
                           f"Expected columns: {required_cols}")

        return df

    def _analyze_distribution(self):
        """Analyze and print class distribution"""
        print("\nClass distribution:")
        for task_id in sorted(self.df['task_id'].unique()):
            count = len(self.df[self.df['task_id'] == task_id])
            info = self.CLASS_INFO.get(task_id, {'name': f'Class{task_id}'})
            strategy = info.get('crop_strategy', 'random')
            print(f"  {info['name']:5s} (task_id={task_id}): {count} samples [crop: {strategy}]")

    def _get_aug_strength(self, task_id: int) -> str:
        """Get augmentation strength for a task"""
        return self.CLASS_INFO.get(task_id, {}).get('aug_strength', 'normal')

    def _get_crop_params(self, task_id: int) -> dict:
        """Get cropping parameters for a task"""
        info = self.CLASS_INFO.get(task_id, {})
        return {
            'strategy': info.get('crop_strategy', 'random'),
            'min_fg': info.get('min_fg', 0.01),
            'max_fg': info.get('max_fg', 0.80),
        }

    def _has_sufficient_tissue(self, image_patch: np.ndarray, min_tissue_ratio: float = 0.10) -> bool:
        """
        Check if patch contains at least min_tissue_ratio of actual tissue.

        White/near-white pixels (RGB all > 240) are considered background (glass slide).
        This ensures we don't train on patches that are mostly empty background.

        Args:
            image_patch: RGB image patch [H, W, 3]
            min_tissue_ratio: Minimum fraction of tissue pixels required (default 10%)

        Returns:
            True if patch has sufficient tissue, False otherwise
        """
        if image_patch.ndim != 3 or image_patch.shape[2] != 3:
            return True  # Can't check, assume valid

        # Detect white background pixels (glass slide)
        # White = all RGB channels > 220 (near white)
        white_threshold = 220
        white_mask = np.all(image_patch > white_threshold, axis=2)

        # Calculate tissue ratio
        total_pixels = white_mask.size
        white_pixels = white_mask.sum()
        tissue_ratio = 1.0 - (white_pixels / total_pixels)

        return tissue_ratio >= min_tissue_ratio

    def _find_object_center(self, mask: np.ndarray) -> Tuple[int, int]:
        """Find the center of mass of foreground objects in mask."""
        # Handle different mask formats
        if mask.ndim == 3:
            binary = (mask[:, :, 0] > 127).astype(np.uint8)
        else:
            binary = (mask > 127).astype(np.uint8)
        
        # Find center of mass
        coords = np.where(binary > 0)
        if len(coords[0]) == 0:
            # No foreground, return image center
            return mask.shape[0] // 2, mask.shape[1] // 2
        
        center_y = int(np.mean(coords[0]))
        center_x = int(np.mean(coords[1]))
        return center_y, center_x

    def _full_structure_crop(self, image: np.ndarray, mask: np.ndarray,
                             crop_size: int = 512, context_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the FULL structure by finding the object bounding box, adding
        context padding, and resizing to crop_size x crop_size.

        This gives the model a complete view of the structure's shape/morphology
        at lower effective resolution. Critical for CAP and TUFT where the full
        structure can span 700-2000+ pixels but crops are only 512x512.

        Args:
            image: Full image [H, W, 3]
            mask: Binary mask [H, W]
            crop_size: Target output size (512)
            context_ratio: How much padding to add around the object (0.2 = 20%)
        """
        h, w = image.shape[:2]

        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        coords = np.where(mask_2d > 127)
        if len(coords[0]) == 0:
            # No foreground - fall back to center crop + resize
            cy, cx = h // 2, w // 2
            half = min(h, w) // 2
            y1, x1 = cy - half, cx - half
            y2, x2 = cy + half, cx + half
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(h, y2), min(w, x2)
            img_crop = image[y1:y2, x1:x2]
            msk_crop = mask_2d[y1:y2, x1:x2]
            img_resized = cv2.resize(img_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            msk_resized = cv2.resize(msk_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
            return img_resized, msk_resized

        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        obj_h = max_y - min_y
        obj_w = max_x - min_x

        # Add context padding around the object
        pad_y = int(obj_h * context_ratio)
        pad_x = int(obj_w * context_ratio)

        # Make the region square (so resize doesn't distort shape)
        region_size = max(obj_h + 2 * pad_y, obj_w + 2 * pad_x)
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2

        # Add slight random jitter to center (so model doesn't always see centered objects)
        jitter = int(region_size * 0.05)
        center_y += np.random.randint(-jitter, jitter + 1)
        center_x += np.random.randint(-jitter, jitter + 1)

        half = region_size // 2
        y1 = center_y - half
        x1 = center_x - half
        y2 = y1 + region_size
        x2 = x1 + region_size

        # Clamp to image bounds
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(h, y2)
        x2 = min(w, x2)

        img_crop = image[y1:y2, x1:x2]
        msk_crop = mask_2d[y1:y2, x1:x2]

        if img_crop.size == 0 or msk_crop.size == 0:
            return image[:crop_size, :crop_size].copy(), mask_2d[:crop_size, :crop_size].copy()

        # Resize to target crop_size (INTER_LINEAR for image, INTER_NEAREST for mask)
        img_resized = cv2.resize(img_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        msk_resized = cv2.resize(msk_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

        return img_resized, msk_resized

    def _object_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, jitter: float = 0.3,
                           max_attempts: int = 30, min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smart cropping for single-object classes (TUFT).

        Uses MIXED SAMPLING strategy with full-structure views:
        - 30% FULL STRUCTURE (entire object resized to 512x512 for shape learning)
        - 10% centered on object (high-res detail)
        - 30% boundary with VARIABLE overlap (diverse FG/BG ratios)
        - 30% random (negative samples / background)

        TISSUE DETECTION: Ensures patches contain at least 10% tissue (not white glass slide).
        """
        h, w = image.shape[:2]

        # Handle different mask formats
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Safety check: if image is smaller than crop_size, return as-is
        # The caller should handle padding
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find object bounding box (do once outside the loop)
        coords = np.where(mask_2d > 127)
        has_fg = len(coords[0]) > 0

        if has_fg:
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            obj_h = max_y - min_y
            obj_w = max_x - min_x
            obj_radius = max(obj_h, obj_w) // 2

        # 30% chance: return full-structure crop (entire object resized)
        if has_fg and np.random.random() < 0.30:
            return self._full_structure_crop(image, mask_2d, crop_size)

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            if not has_fg:
                # No foreground, random crop
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))
            else:
                # Randomly choose sampling strategy (remaining 70%)
                strategy = np.random.random()

                if strategy < 0.14:
                    # Strategy 1: CENTERED on object (10% of total = 14% of remaining 70%)
                    jitter_y = int(np.random.uniform(-jitter, jitter) * crop_size)
                    jitter_x = int(np.random.uniform(-jitter, jitter) * crop_size)
                    half = crop_size // 2
                    y1 = center_y - half + jitter_y
                    x1 = center_x - half + jitter_x

                elif strategy < 0.57:
                    # Strategy 2: BOUNDARY with VARIABLE overlap (30% of total = 43% of remaining 70%)
                    angle = np.random.uniform(0, 2 * np.pi)
                    overlap_zone = np.random.random()

                    if overlap_zone < 0.33:
                        distance = obj_radius * 0.3 + crop_size // 4
                    elif overlap_zone < 0.66:
                        distance = obj_radius + crop_size // 4
                    else:
                        distance = obj_radius + crop_size // 2.5

                    crop_center_y = center_y + int(distance * np.sin(angle))
                    crop_center_x = center_x + int(distance * np.cos(angle))
                    half = crop_size // 2
                    y1 = crop_center_y - half
                    x1 = crop_center_x - half

                else:
                    # Strategy 3: RANDOM position (30% of total = 43% of remaining 70%)
                    y1 = np.random.randint(0, max(1, h - crop_size + 1))
                    x1 = np.random.randint(0, max(1, w - crop_size + 1))

            # Clamp to valid image bounds
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _glomerular_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                               crop_size: int = 512, jitter: float = 0.15,
                               max_attempts: int = 30, min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Glomerular-aware cropping for CAP (Bowman's capsule).

        Designed to capture full glomeruli with complete capsule boundaries.
        Includes full-structure crops at lower resolution so the model learns
        the complete capsule shape, not just fragments.

        Strategy:
        - 30% FULL STRUCTURE (entire capsule resized to 512x512)
        - 15% CENTERED (tight jitter=0.15, high-res detail)
        - 25% BOUNDARY (variable overlap for boundary diversity)
        - 30% RANDOM (negative samples / background)

        TISSUE DETECTION: Ensures patches contain at least 10% tissue.
        """
        h, w = image.shape[:2]

        # Handle different mask formats
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Safety check
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find object bounding box
        coords = np.where(mask_2d > 127)
        has_fg = len(coords[0]) > 0

        if has_fg:
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            obj_h = max_y - min_y
            obj_w = max_x - min_x
            obj_radius = max(obj_h, obj_w) // 2

        # 30% chance: return full-structure crop (entire capsule resized)
        if has_fg and np.random.random() < 0.30:
            return self._full_structure_crop(image, mask_2d, crop_size)

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            if not has_fg:
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))
            else:
                strategy = np.random.random()

                if strategy < 0.21:
                    # 15% of total = 21% of remaining 70%: CENTERED
                    jitter_y = int(np.random.uniform(-jitter, jitter) * crop_size)
                    jitter_x = int(np.random.uniform(-jitter, jitter) * crop_size)
                    half = crop_size // 2
                    y1 = center_y - half + jitter_y
                    x1 = center_x - half + jitter_x

                elif strategy < 0.57:
                    # 25% of total = 36% of remaining 70%: BOUNDARY
                    angle = np.random.uniform(0, 2 * np.pi)
                    overlap_zone = np.random.random()

                    if overlap_zone < 0.33:
                        distance = obj_radius * 0.3 + crop_size // 4
                    elif overlap_zone < 0.66:
                        distance = obj_radius + crop_size // 4
                    else:
                        distance = obj_radius + crop_size // 2.5

                    crop_center_y = center_y + int(distance * np.sin(angle))
                    crop_center_x = center_x + int(distance * np.cos(angle))
                    half = crop_size // 2
                    y1 = crop_center_y - half
                    x1 = crop_center_x - half

                else:
                    # 30% of total = 43% of remaining 70%: RANDOM
                    y1 = np.random.randint(0, max(1, h - crop_size + 1))
                    x1 = np.random.randint(0, max(1, w - crop_size + 1))

            # Clamp to valid image bounds
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        return image.copy(), mask_2d.copy()

    def _tubule_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, max_attempts: int = 30,
                           min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smart cropping for multi-instance structures (DT, PT tubules).

        Uses MIXED SAMPLING strategy for better generalization:
        - 30% foreground-centered (sample from tubule pixels)
        - 30% boundary (partial tubule coverage)
        - 40% random/background (negative samples)

        TISSUE DETECTION: Ensures patches contain at least 10% tissue (not white glass slide).

        Increased random sampling to reduce overfitting on centered patches.
        This teaches the model to:
        - Recognize tubule structures at various positions
        - Handle partial tubules at boundaries
        - Correctly predict background (prevent false positives)
        """
        h, w = image.shape[:2]

        # Handle different mask formats
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Safety check: if image is smaller than crop_size, return as-is
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find foreground pixels (do once outside the loop)
        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 100  # Need at least some foreground

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            # Randomly choose sampling strategy
            strategy = np.random.random()

            if not has_fg or strategy > 0.60:
                # 40%: RANDOM/BACKGROUND (or no FG available)
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))

            elif strategy > 0.30:
                # 30%: BOUNDARY - partial coverage
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                # Large offset in random direction to get partial coverage
                offset = crop_size // 2
                angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            else:
                # 30%: FOREGROUND-CENTERED
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                # Small jitter for variety
                jitter = int(crop_size * 0.15)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            # Clamp to valid range
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _sparse_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, max_attempts: int = 30,
                           min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sparse-aware cropping for moderately sparse structures like PT.

        Uses HIGH RANDOM sampling to reflect natural distribution:
        - 70% RANDOM (reflects that most patches may not contain the structure)
        - 20% FG-centered (ensure some positive samples)
        - 10% boundary (partial coverage)

        TISSUE DETECTION: Ensures patches contain at least 10% tissue (not white glass slide).

        This is less biased than tubule_aware, better for PT which has ~30-40% coverage.
        """
        h, w = image.shape[:2]

        # Handle different mask formats
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Safety check
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find foreground pixels (do once outside the loop)
        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 100

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            # Randomly choose sampling strategy
            strategy = np.random.random()

            if not has_fg or strategy > 0.30:
                # 70%: RANDOM - reflects natural sparse distribution
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))

            elif strategy > 0.10:
                # 20%: FG-CENTERED with jitter
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                # Moderate jitter
                jitter = int(crop_size * 0.2)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            else:
                # 10%: BOUNDARY - partial coverage
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                offset = crop_size // 2
                angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            # Clamp to valid range
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _dt_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                       crop_size: int = 512, max_attempts: int = 30,
                       min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        DT-specific cropping for distal tubules.

        Uses BALANCED sampling strategy:
        - 30% FOREGROUND-CENTERED (sample from tubule pixels with jitter)
        - 30% BOUNDARY (partial tubule coverage for better edge learning)
        - 40% RANDOM (background/negative samples)

        TISSUE DETECTION: Ensures patches contain at least 10% tissue (not white glass slide).

        Jitter: ±15% of crop_size = ±76 pixels for centered patches.
        """
        h, w = image.shape[:2]

        # Handle different mask formats
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Safety check
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find foreground pixels (do once outside the loop)
        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 100

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            # Randomly choose sampling strategy
            strategy = np.random.random()

            if not has_fg or strategy > 0.60:
                # 40%: RANDOM (or no FG available)
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))

            elif strategy > 0.30:
                # 30%: BOUNDARY - partial coverage for better edge learning
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                # Offset by crop_size//2 in random direction
                offset = crop_size // 2
                angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            else:
                # 30%: FOREGROUND-CENTERED
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                # Jitter: ±15% of crop_size = ±76 pixels
                jitter = int(crop_size * 0.15)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            # Clamp to valid range
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _smart_random_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, min_fg: float = 0.01,
                           max_fg: float = 0.80, max_attempts: int = 50,
                           min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random crop with foreground constraints.

        Rejects patches that have too little or too much foreground.
        TISSUE DETECTION: Also ensures patches contain at least 10% tissue (not white glass slide).
        """
        h, w = image.shape[:2]

        # Handle different mask formats
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Safety check: if image is smaller than crop_size, return as-is
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        best_crop = None
        best_score = 0.0  # Combined score considering both FG ratio and tissue

        for attempt in range(max_attempts):
            # Random crop position - ensure valid range
            y1 = np.random.randint(0, max(1, h - crop_size + 1))
            x1 = np.random.randint(0, max(1, w - crop_size + 1))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Get patches
            image_crop = image[y1:y2, x1:x2]
            mask_patch = mask_2d[y1:y2, x1:x2]

            if mask_patch.size == 0 or image_crop.size == 0:
                continue

            # Compute foreground ratio
            fg_ratio = np.sum(mask_patch > 127) / (crop_size * crop_size)

            # Check tissue ratio
            has_tissue = self._has_sufficient_tissue(image_crop, min_tissue_ratio)

            # Check both constraints
            if min_fg <= fg_ratio <= max_fg and has_tissue:
                # Good patch found - meets both criteria
                if mask.ndim == 3:
                    mask_crop = mask[y1:y2, x1:x2]
                else:
                    mask_crop = mask_patch
                return image_crop, mask_crop

            # Track best patch for fallback (prefer high tissue + acceptable FG)
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)

            # Score: prioritize tissue, then FG ratio
            score = tissue_ratio * 0.7 + (fg_ratio if fg_ratio <= max_fg else 0) * 0.3
            if score > best_score:
                best_score = score
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found or center crop
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
        else:
            y1 = max(0, (h - crop_size) // 2)
            x1 = max(0, (w - crop_size) // 2)
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

        image_crop = image[y1:y2, x1:x2]
        if mask.ndim == 3:
            mask_crop = mask[y1:y2, x1:x2]
        else:
            mask_crop = mask_2d[y1:y2, x1:x2]

        # Safety check: ensure output is not empty
        if image_crop.size == 0 or mask_crop.size == 0:
            return image.copy(), mask_2d.copy()

        return image_crop, mask_crop

    def _vessel_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, max_attempts: int = 30,
                           min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Axis-aware vessel cropping for ART/VES class (large elongated structures).

        Based on pixel analysis:
        - VES has ~3.35% coverage (sparse)
        - 2-3 LARGE objects per image (~120,855 pixels average)
        - Objects are elongated tubular structures

        Uses AXIS-AWARE sampling to exploit vessel orientation:
        - 30% FOREGROUND-CENTERED (sample from vessel pixels with jitter)
        - 30% AXIS-BOUNDARY (sample along/perpendicular to vessel axis)
        - 40% RANDOM (background/negative samples)

        The axis-aware boundary sampling uses PCA on foreground pixels to find
        the vessel's principal axis, then samples perpendicular to it for
        cross-section views and along it for longitudinal views.

        TISSUE DETECTION: Ensures patches contain at least 10% tissue.
        """
        h, w = image.shape[:2]

        # Handle different mask formats
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Safety check
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find foreground pixels
        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 100

        # Compute vessel axis via PCA if we have enough foreground
        vessel_angle = None
        if has_fg and len(fg_coords[0]) > 500:
            try:
                # Subsample for speed
                n_pts = min(len(fg_coords[0]), 5000)
                indices = np.random.choice(len(fg_coords[0]), n_pts, replace=False)
                pts = np.stack([fg_coords[1][indices], fg_coords[0][indices]], axis=1).astype(np.float64)  # x, y
                pts -= pts.mean(axis=0)
                cov = np.cov(pts.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Principal axis is the eigenvector with largest eigenvalue
                principal = eigenvectors[:, -1]  # (dx, dy)
                vessel_angle = np.arctan2(principal[1], principal[0])
            except Exception:
                vessel_angle = None

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            strategy = np.random.random()

            if not has_fg or strategy > 0.60:
                # 40%: RANDOM (or no FG available)
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))

            elif strategy > 0.30:
                # 30%: AXIS-BOUNDARY - sample along/perpendicular to vessel
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                offset = crop_size // 2
                if vessel_angle is not None:
                    # 50% perpendicular (cross-section), 50% along axis (longitudinal)
                    if np.random.random() < 0.5:
                        angle = vessel_angle + np.pi / 2  # perpendicular
                    else:
                        angle = vessel_angle  # along axis
                    # Add small random perturbation (±20 degrees)
                    angle += np.random.uniform(-0.35, 0.35)
                else:
                    # Fallback: random angle
                    angle = np.random.uniform(0, 2 * np.pi)

                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            else:
                # 30%: FOREGROUND-CENTERED
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                # Jitter: ±20% of crop_size for variety
                jitter = int(crop_size * 0.20)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            # Clamp to valid range
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _ptc_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                        crop_size: int = 512, max_attempts: int = 50,
                        min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        PTC-aware cropping for Peritubular Capillaries (many tiny scattered objects).

        Based on pixel analysis:
        - PTC has ~3.62% coverage (sparse)
        - 171 TINY objects per image (~1,905 pixels average = ~0.7% of a patch)
        - Objects are scattered throughout the image

        Uses BALANCED sampling strategy (reduced FG bias for better val match):
        - 40% FOREGROUND-CENTERED (sample from tiny capillary pixels)
        - 20% BOUNDARY (partial coverage for edge learning)
        - 40% RANDOM (background/negative samples — matches val sliding window)

        Key difference: Lower has_fg threshold (50 pixels instead of 100) because
        PTC objects are very small. Also uses 50 attempts instead of 30.

        TISSUE DETECTION: Ensures patches contain at least 10% tissue.
        """
        h, w = image.shape[:2]

        # Handle different mask formats
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Safety check
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find foreground pixels - lower threshold for tiny objects
        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 50  # Lower threshold for tiny PTC objects

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            strategy = np.random.random()

            if not has_fg or strategy > 0.60:
                # 40%: RANDOM (or no FG available) — better matches val distribution
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))

            elif strategy > 0.40:
                # 20%: BOUNDARY - partial coverage
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                # Smaller offset for tiny objects
                offset = crop_size // 3
                angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            else:
                # 40%: FOREGROUND-CENTERED (reduced from 50% for better val match)
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]

                # Small jitter to keep tiny objects in view
                jitter = int(crop_size * 0.10)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)

                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            # Clamp to valid range
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def __len__(self) -> int:
        if self.max_iters:
            return self.max_iters
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        """
        Get a training sample.

        Returns:
            Dictionary with:
                - image: [3, H, W] normalized image
                - label: [H, W] binary mask
                - weight: [H, W] edge weight map
                - name: sample identifier
                - task_id: class/task ID
                - scale_id: scale ID
        """
        # Handle oversampling
        if self.max_iters:
            index = index % len(self.df)

        # Get data row
        row = self.df.iloc[index]

        # Load image and mask
        image = np.array(Image.open(row['image_path']).convert('RGB'))
        mask = np.array(Image.open(row['label_path']).convert('RGB'))

        # Get metadata
        name = row['name']
        task_id = int(row['task_id'])
        scale_id = int(row['scale_id'])

        # Get crop parameters for this class
        crop_params = self._get_crop_params(task_id)

        # Ensure mask is single channel for processing
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        h, w = image.shape[:2]

        # Smart cropping based on class type
        if h > 512 or w > 512:
            if crop_params['strategy'] == 'glomerular_aware':
                # Glomerular-aware cropping for CAP (more centered for full capsule boundary)
                # 25% centered, 35% boundary, 40% random
                image, mask = self._glomerular_aware_crop(
                    image, mask,
                    crop_size=512,
                    jitter=0.15
                )
            elif crop_params['strategy'] == 'object_aware':
                # Object-aware cropping for TUFT (single-object classes)
                image, mask = self._object_aware_crop(
                    image, mask,
                    crop_size=512,
                    jitter=0.3
                )
            elif crop_params['strategy'] == 'tubule_aware':
                # Tubule-aware cropping for DT/PT (multi-instance classes)
                # Mixed sampling: 30% FG-centered, 30% boundary, 40% background
                image, mask = self._tubule_aware_crop(
                    image, mask,
                    crop_size=512
                )
            elif crop_params['strategy'] == 'sparse_aware':
                # Sparse-aware cropping for PT (moderate density)
                # 70% random + 30% FG-aware (better for moderately sparse structures)
                image, mask = self._sparse_aware_crop(
                    image, mask,
                    crop_size=512
                )
            elif crop_params['strategy'] == 'dt_aware':
                # DT-specific cropping: 30% centered, 20% boundary, 50% random
                image, mask = self._dt_aware_crop(
                    image, mask,
                    crop_size=512
                )
            elif crop_params['strategy'] == 'vessel_aware':
                # VES: Vessel-aware cropping for large elongated structures
                # 30% FG-centered, 30% boundary, 40% random
                image, mask = self._vessel_aware_crop(
                    image, mask,
                    crop_size=512
                )
            elif crop_params['strategy'] == 'ptc_aware':
                # PTC: Balanced sampling for tiny scattered capillaries
                # 40% FG-centered, 20% boundary, 40% random (better val match)
                image, mask = self._ptc_aware_crop(
                    image, mask,
                    crop_size=512
                )
            else:
                # Fallback: Smart random cropping with foreground constraints
                image, mask = self._smart_random_crop(
                    image, mask,
                    crop_size=512,
                    min_fg=crop_params['min_fg'],
                    max_fg=crop_params['max_fg'],
                    max_attempts=50
                )
        elif h < 512 or w < 512:
            # Pad small images
            if HAS_IMGAUG:
                image = np.expand_dims(image, axis=0)
                mask = np.expand_dims(mask, axis=0)
                image, mask = self.pad_512(
                    images=image, 
                    heatmaps=np.expand_dims(mask.astype(np.float32) / 255.0, -1)
                )
                mask = (mask[:, :, :, 0] * 255).astype(np.uint8)
                image = image[0]
                mask = mask[0]

        # Add batch dimension for imgaug augmentation
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Apply augmentation
        if self.augment and self.transforms:
            aug_strength = self._get_aug_strength(task_id)
            transforms = self.transforms.get(aug_strength, self.transforms['normal'])

            if transforms:
                seed = np.random.rand(4)

                # Geometric augmentation (applied to both image and mask)
                if seed[0] > 0.5:
                    image, mask = transforms['geometric'](
                        images=image,
                        heatmaps=np.expand_dims(mask.astype(np.float32) / 255.0, -1)
                    )
                    mask = (mask[:, :, :, 0] * 255).astype(np.uint8)

                # Color augmentation (image only)
                if seed[1] > 0.5:
                    image = transforms['color'](images=image)

                # Noise augmentation (image only)
                if seed[2] > 0.5:
                    image = transforms['noise'](images=image)

        # Remove batch dimension
        image = image[0]
        mask = mask[0]

        # Always ensure final size is exactly 512x512
        target_size = 512
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape[0] != target_size or mask.shape[1] != target_size:
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        # Final assertion to catch any remaining issues
        assert image.shape[:2] == (target_size, target_size), f"Image shape {image.shape} != ({target_size}, {target_size})"
        assert mask.shape == (target_size, target_size), f"Mask shape {mask.shape} != ({target_size}, {target_size})"

        # Binarize mask
        mask = (mask >= 127).astype(np.float32)

        # Compute edge weight
        if self.edge_weight:
            weight = self._compute_edge_weight(mask)
        else:
            weight = np.ones_like(mask)

        # Normalize and convert to tensor format
        image = image.transpose(2, 0, 1).astype(np.float32)  # CHW
        if image.max() > 1:
            image = image / 255.0

        return {
            'image': image,
            'label': mask.astype(np.float32),
            'weight': weight.astype(np.float32),
            'name': name,
            'task_id': task_id,
            'scale_id': scale_id
        }

    def _compute_edge_weight(self, mask: np.ndarray, iterations: int = 2) -> np.ndarray:
        """Compute edge weight map for boundary-aware loss"""
        dilated = scipy.ndimage.binary_dilation(mask > 0.5, iterations=iterations)
        edge = dilated & ~(mask > 0.5)
        weight = np.ones_like(mask, dtype=np.float32)
        weight[edge] = 2.0  # Higher weight on edges
        return weight


class PathologyValDataset(Dataset):
    """
    Validation dataset with the SAME cropping strategies as training.

    Uses IDENTICAL crop strategies as PathologyDataset:
    - random for DT (very sparse ~5-10% coverage)
    - sparse_aware for PT (moderate ~30-40% coverage)
    - object_aware for CAP/TUFT (single dense objects)

    Only difference from training: NO augmentation applied.
    This ensures train/val use the same patch distribution.
    """

    # Class information with cropping strategy (MUST MATCH PathologyDataset!)
    CLASS_INFO = {
        0: {'name': 'DT', 'scale': 1, 'samples': 72, 'crop_strategy': 'dt_aware',  # 30% centered, 30% boundary, 40% random
            'min_fg': 0.01, 'max_fg': 0.70},
        1: {'name': 'PT', 'scale': 1, 'samples': 74, 'crop_strategy': 'sparse_aware',  # sparse_aware
            'min_fg': 0.01, 'max_fg': 0.70},
        2: {'name': 'CAP', 'scale': 0, 'samples': 94, 'crop_strategy': 'glomerular_aware',  # 25% centered, 35% boundary, 40% random
            'min_fg': 0.02, 'max_fg': 0.95},
        3: {'name': 'TUFT', 'scale': 0, 'samples': 99, 'crop_strategy': 'object_aware',
            'min_fg': 0.02, 'max_fg': 0.95},
        4: {'name': 'VES', 'scale': 1, 'samples': 46, 'crop_strategy': 'vessel_aware',  # 30% FG, 30% axis-boundary, 40% random
            'min_fg': 0.01, 'max_fg': 0.80},
        5: {'name': 'PTC', 'scale': 3, 'samples': 25, 'crop_strategy': 'ptc_aware',  # 40% FG, 20% boundary, 40% random
            'min_fg': 0.001, 'max_fg': 0.80}
    }

    def __init__(
        self,
        csv_path: str,
        target_size: Tuple[int, int] = (512, 512),
        exclude_classes: Optional[List[int]] = None
    ):
        super().__init__()

        self.csv_path = csv_path
        self.target_h, self.target_w = target_size
        self.exclude_classes = exclude_classes or []

        self.df = self._load_csv(csv_path)

        # Filter out excluded classes
        if self.exclude_classes:
            original_len = len(self.df)
            self.df = self.df[~self.df['task_id'].isin(self.exclude_classes)].reset_index(drop=True)
            excluded_names = [self.CLASS_INFO.get(c, {}).get('name', f'Class{c}') for c in self.exclude_classes]
            print(f"Excluded classes {excluded_names}: {original_len} -> {len(self.df)} validation samples")

        print(f"Loaded {len(self.df)} validation samples")

        if HAS_IMGAUG:
            self.pad = iaa.PadToFixedSize(
                width=target_size[1],
                height=target_size[0],
                position='center'
            )

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with auto-detection of separator and column names"""
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                required_cols = ['image_path', 'label_path', 'task_id', 'scale_id']
                if all(col in df.columns for col in required_cols):
                    return df
            except Exception:
                continue

        df = pd.read_csv(csv_path, sep=None, engine='python')

        column_mapping = {
            'img_path': 'image_path',
            'image': 'image_path',
            'mask_path': 'label_path',
            'mask': 'label_path',
            'label': 'label_path',
            'class_id': 'task_id',
            'class': 'task_id',
            'task': 'task_id',
            'scale': 'scale_id',
            'mag': 'scale_id',
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        return df

    def _get_crop_params(self, task_id: int) -> dict:
        """Get cropping parameters for a task"""
        info = self.CLASS_INFO.get(task_id, {})
        return {
            'strategy': info.get('crop_strategy', 'random'),
            'min_fg': info.get('min_fg', 0.01),
            'max_fg': info.get('max_fg', 0.80),
        }

    def _has_sufficient_tissue(self, image_patch: np.ndarray, min_tissue_ratio: float = 0.10) -> bool:
        """
        Check if patch contains at least min_tissue_ratio of actual tissue.

        White/near-white pixels (RGB all > 240) are considered background (glass slide).
        This ensures we don't validate on patches that are mostly empty background.

        Args:
            image_patch: RGB image patch [H, W, 3]
            min_tissue_ratio: Minimum fraction of tissue pixels required (default 10%)

        Returns:
            True if patch has sufficient tissue, False otherwise
        """
        if image_patch.ndim != 3 or image_patch.shape[2] != 3:
            return True  # Can't check, assume valid

        # Detect white background pixels (glass slide)
        white_threshold = 220
        white_mask = np.all(image_patch > white_threshold, axis=2)

        # Calculate tissue ratio
        total_pixels = white_mask.size
        white_pixels = white_mask.sum()
        tissue_ratio = 1.0 - (white_pixels / total_pixels)

        return tissue_ratio >= min_tissue_ratio

    def _full_structure_crop(self, image: np.ndarray, mask: np.ndarray,
                             crop_size: int = 512, context_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the FULL structure by finding the object bounding box, adding
        context padding, and resizing to crop_size x crop_size.
        SAME as training. Gives model complete view of structure shape.
        """
        h, w = image.shape[:2]

        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        coords = np.where(mask_2d > 127)
        if len(coords[0]) == 0:
            cy, cx = h // 2, w // 2
            half = min(h, w) // 2
            y1, x1 = max(0, cy - half), max(0, cx - half)
            y2, x2 = min(h, cy + half), min(w, cx + half)
            img_crop = image[y1:y2, x1:x2]
            msk_crop = mask_2d[y1:y2, x1:x2]
            img_resized = cv2.resize(img_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            msk_resized = cv2.resize(msk_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
            return img_resized, msk_resized

        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        obj_h = max_y - min_y
        obj_w = max_x - min_x

        pad_y = int(obj_h * context_ratio)
        pad_x = int(obj_w * context_ratio)

        region_size = max(obj_h + 2 * pad_y, obj_w + 2 * pad_x)
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2

        jitter = int(region_size * 0.05)
        center_y += np.random.randint(-jitter, jitter + 1)
        center_x += np.random.randint(-jitter, jitter + 1)

        half = region_size // 2
        y1 = max(0, center_y - half)
        x1 = max(0, center_x - half)
        y2 = min(h, y1 + region_size)
        x2 = min(w, x1 + region_size)

        img_crop = image[y1:y2, x1:x2]
        msk_crop = mask_2d[y1:y2, x1:x2]

        if img_crop.size == 0 or msk_crop.size == 0:
            return image[:crop_size, :crop_size].copy(), mask_2d[:crop_size, :crop_size].copy()

        img_resized = cv2.resize(img_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        msk_resized = cv2.resize(msk_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

        return img_resized, msk_resized

    def _object_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, jitter: float = 0.3,
                           max_attempts: int = 30, min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smart cropping for single-object classes (TUFT).
        SAME mixed sampling strategy as training:
        - 30% FULL STRUCTURE (entire object resized to 512x512)
        - 10% centered on object (high-res detail)
        - 30% boundary with variable overlap
        - 30% random (background)

        TISSUE DETECTION: Ensures patches contain at least 10% tissue (not white glass slide).
        """
        h, w = image.shape[:2]

        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find object bounding box (do once outside the loop)
        coords = np.where(mask_2d > 127)
        has_fg = len(coords[0]) > 0

        if has_fg:
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            obj_h = max_y - min_y
            obj_w = max_x - min_x
            obj_radius = max(obj_h, obj_w) // 2

        # 30% chance: return full-structure crop (entire object resized)
        if has_fg and np.random.random() < 0.30:
            return self._full_structure_crop(image, mask_2d, crop_size)

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            if not has_fg:
                # No foreground, random crop
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))
            else:
                # Remaining 70% split into centered/boundary/random
                strategy = np.random.random()

                if strategy < 0.14:
                    # 10% of total = 14% of remaining 70%: CENTERED
                    jitter_y = int(np.random.uniform(-jitter, jitter) * crop_size)
                    jitter_x = int(np.random.uniform(-jitter, jitter) * crop_size)
                    half = crop_size // 2
                    y1 = center_y - half + jitter_y
                    x1 = center_x - half + jitter_x

                elif strategy < 0.57:
                    # 30% of total = 43% of remaining 70%: BOUNDARY
                    angle = np.random.uniform(0, 2 * np.pi)
                    overlap_zone = np.random.random()

                    if overlap_zone < 0.33:
                        distance = obj_radius * 0.3 + crop_size // 4
                    elif overlap_zone < 0.66:
                        distance = obj_radius + crop_size // 4
                    else:
                        distance = obj_radius + crop_size // 2.5

                    crop_center_y = center_y + int(distance * np.sin(angle))
                    crop_center_x = center_x + int(distance * np.cos(angle))
                    half = crop_size // 2
                    y1 = crop_center_y - half
                    x1 = crop_center_x - half

                else:
                    # 30% of total = 43% of remaining 70%: RANDOM
                    y1 = np.random.randint(0, max(1, h - crop_size + 1))
                    x1 = np.random.randint(0, max(1, w - crop_size + 1))

            # Clamp to valid bounds
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _tubule_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, max_attempts: int = 30,
                           min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smart cropping for multi-instance structures (DT, PT tubules).
        SAME mixed sampling strategy as training:
        - 30% foreground-centered
        - 30% boundary (partial tubule)
        - 40% random/background

        TISSUE DETECTION: Ensures patches contain at least 10% tissue (not white glass slide).
        """
        h, w = image.shape[:2]

        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find foreground pixels (do once outside the loop)
        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 100

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            strategy = np.random.random()

            if not has_fg or strategy > 0.60:
                # Random/background (40%)
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))

            elif strategy > 0.30:
                # Boundary - partial coverage (30%)
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                offset = crop_size // 2
                angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            else:
                # Foreground-centered (30%)
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                jitter = int(crop_size * 0.15)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            # Clamp to valid range
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _sparse_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, max_attempts: int = 30,
                           min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sparse-aware cropping for PT (moderate density ~30-40%).
        SAME as training: 70% random, 20% FG-centered, 10% boundary.

        TISSUE DETECTION: Ensures patches contain at least 10% tissue (not white glass slide).
        """
        h, w = image.shape[:2]

        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find foreground pixels (do once outside the loop)
        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 100

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            strategy = np.random.random()

            if not has_fg or strategy > 0.30:
                # 70%: RANDOM
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))
            elif strategy > 0.10:
                # 20%: FG-CENTERED
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                jitter = int(crop_size * 0.2)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2
            else:
                # 10%: BOUNDARY
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                offset = crop_size // 2
                angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            # Clamp to valid range
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _dt_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                       crop_size: int = 512, max_attempts: int = 30,
                       min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        DT-specific cropping for distal tubules.
        SAME as training: 30% centered, 30% boundary, 40% random.

        TISSUE DETECTION: Ensures patches contain at least 10% tissue (not white glass slide).
        Jitter: ±15% of crop_size = ±76 pixels for centered patches.
        """
        h, w = image.shape[:2]

        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        # Find foreground pixels (do once outside the loop)
        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 100

        best_crop = None
        best_tissue_ratio = 0.0

        for attempt in range(max_attempts):
            strategy = np.random.random()

            if not has_fg or strategy > 0.60:
                # 40%: RANDOM
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))
            elif strategy > 0.30:
                # 30%: BOUNDARY - for better edge learning
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                offset = crop_size // 2
                angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2
            else:
                # 30%: FOREGROUND-CENTERED
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                jitter = int(crop_size * 0.15)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            # Clamp to valid range
            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Crop
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]

            if image_crop.size == 0 or mask_crop.size == 0:
                continue

            # Check tissue ratio
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop

            # Track best patch for fallback
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]

        # Ultimate fallback
        return image.copy(), mask_2d.copy()

    def _smart_random_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, min_fg: float = 0.01,
                           max_fg: float = 0.80, max_attempts: int = 50,
                           min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random crop with foreground constraints.
        SAME logic as training - rejects patches with too little or too much foreground.

        TISSUE DETECTION: Also ensures patches contain at least 10% tissue (not white glass slide).
        """
        h, w = image.shape[:2]

        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        best_crop = None
        best_score = 0.0  # Combined score considering both FG ratio and tissue

        for attempt in range(max_attempts):
            y1 = np.random.randint(0, max(1, h - crop_size + 1))
            x1 = np.random.randint(0, max(1, w - crop_size + 1))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

            # Get patches
            image_crop = image[y1:y2, x1:x2]
            mask_patch = mask_2d[y1:y2, x1:x2]

            if mask_patch.size == 0 or image_crop.size == 0:
                continue

            # Compute foreground ratio
            fg_ratio = np.sum(mask_patch > 127) / (crop_size * crop_size)

            # Check tissue ratio
            has_tissue = self._has_sufficient_tissue(image_crop, min_tissue_ratio)

            # Check both constraints
            if min_fg <= fg_ratio <= max_fg and has_tissue:
                # Good patch found - meets both criteria
                if mask.ndim == 3:
                    mask_crop = mask[y1:y2, x1:x2]
                else:
                    mask_crop = mask_patch
                return image_crop, mask_crop

            # Track best patch for fallback (prefer high tissue + acceptable FG)
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)

            # Score: prioritize tissue, then FG ratio
            score = tissue_ratio * 0.7 + (fg_ratio if fg_ratio <= max_fg else 0) * 0.3
            if score > best_score:
                best_score = score
                best_crop = (y1, x1, y2, x2)

        # Fallback: use best patch found or center crop
        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
        else:
            y1 = max(0, (h - crop_size) // 2)
            x1 = max(0, (w - crop_size) // 2)
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)

        image_crop = image[y1:y2, x1:x2]
        if mask.ndim == 3:
            mask_crop = mask[y1:y2, x1:x2]
        else:
            mask_crop = mask_2d[y1:y2, x1:x2]

        # Safety check: ensure output is not empty
        if image_crop.size == 0 or mask_crop.size == 0:
            return image.copy(), mask_2d.copy()

        return image_crop, mask_crop

    def _glomerular_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                               crop_size: int = 512, jitter: float = 0.15,
                               max_attempts: int = 30, min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Glomerular-aware cropping for CAP. SAME as training.
        30% FULL STRUCTURE, 15% centered, 25% boundary, 30% random.
        """
        h, w = image.shape[:2]
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        coords = np.where(mask_2d > 127)
        has_fg = len(coords[0]) > 0
        if has_fg:
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            obj_h = max_y - min_y
            obj_w = max_x - min_x
            obj_radius = max(obj_h, obj_w) // 2

        # 30% chance: return full-structure crop (entire capsule resized)
        if has_fg and np.random.random() < 0.30:
            return self._full_structure_crop(image, mask_2d, crop_size)

        best_crop = None
        best_tissue_ratio = 0.0
        for attempt in range(max_attempts):
            if not has_fg:
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))
            else:
                strategy = np.random.random()
                if strategy < 0.21:
                    # 15% of total = 21% of remaining 70%: CENTERED
                    jitter_y = int(np.random.uniform(-jitter, jitter) * crop_size)
                    jitter_x = int(np.random.uniform(-jitter, jitter) * crop_size)
                    half = crop_size // 2
                    y1 = center_y - half + jitter_y
                    x1 = center_x - half + jitter_x
                elif strategy < 0.57:
                    # 25% of total = 36% of remaining 70%: BOUNDARY
                    angle = np.random.uniform(0, 2 * np.pi)
                    overlap_zone = np.random.random()
                    if overlap_zone < 0.33:
                        distance = obj_radius * 0.3 + crop_size // 4
                    elif overlap_zone < 0.66:
                        distance = obj_radius + crop_size // 4
                    else:
                        distance = obj_radius + crop_size // 2.5
                    crop_center_y = center_y + int(distance * np.sin(angle))
                    crop_center_x = center_x + int(distance * np.cos(angle))
                    half = crop_size // 2
                    y1 = crop_center_y - half
                    x1 = crop_center_x - half
                else:
                    # 30% of total = 43% of remaining 70%: RANDOM
                    y1 = np.random.randint(0, max(1, h - crop_size + 1))
                    x1 = np.random.randint(0, max(1, w - crop_size + 1))

            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]
            if image_crop.size == 0 or mask_crop.size == 0:
                continue
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]
        return image.copy(), mask_2d.copy()

    def _vessel_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                           crop_size: int = 512, max_attempts: int = 30,
                           min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Axis-aware vessel cropping for ART/VES class.
        SAME as training: 30% FG-centered, 30% axis-boundary, 40% random.
        Uses PCA to detect vessel orientation for smarter boundary sampling.
        """
        h, w = image.shape[:2]
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 100

        # Compute vessel axis via PCA
        vessel_angle = None
        if has_fg and len(fg_coords[0]) > 500:
            try:
                n_pts = min(len(fg_coords[0]), 5000)
                indices = np.random.choice(len(fg_coords[0]), n_pts, replace=False)
                pts = np.stack([fg_coords[1][indices], fg_coords[0][indices]], axis=1).astype(np.float64)
                pts -= pts.mean(axis=0)
                cov = np.cov(pts.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                principal = eigenvectors[:, -1]
                vessel_angle = np.arctan2(principal[1], principal[0])
            except Exception:
                vessel_angle = None

        best_crop = None
        best_tissue_ratio = 0.0
        for attempt in range(max_attempts):
            strategy = np.random.random()
            if not has_fg or strategy > 0.60:
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))
            elif strategy > 0.30:
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                offset = crop_size // 2
                if vessel_angle is not None:
                    if np.random.random() < 0.5:
                        angle = vessel_angle + np.pi / 2
                    else:
                        angle = vessel_angle
                    angle += np.random.uniform(-0.35, 0.35)
                else:
                    angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2
            else:
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                jitter = int(crop_size * 0.20)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]
            if image_crop.size == 0 or mask_crop.size == 0:
                continue
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]
        return image.copy(), mask_2d.copy()

    def _ptc_aware_crop(self, image: np.ndarray, mask: np.ndarray,
                        crop_size: int = 512, max_attempts: int = 50,
                        min_tissue_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        PTC-aware cropping for tiny scattered capillaries.
        SAME as training: 40% FG-centered, 20% boundary, 40% random.
        """
        h, w = image.shape[:2]
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        if h < crop_size or w < crop_size:
            return image.copy(), mask_2d.copy()

        fg_coords = np.where(mask_2d > 127)
        has_fg = len(fg_coords[0]) > 50

        best_crop = None
        best_tissue_ratio = 0.0
        for attempt in range(max_attempts):
            strategy = np.random.random()
            if not has_fg or strategy > 0.60:
                # 40%: RANDOM
                y1 = np.random.randint(0, max(1, h - crop_size + 1))
                x1 = np.random.randint(0, max(1, w - crop_size + 1))
            elif strategy > 0.40:
                # 20%: BOUNDARY
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                offset = crop_size // 3
                angle = np.random.uniform(0, 2 * np.pi)
                cy += int(offset * np.sin(angle))
                cx += int(offset * np.cos(angle))
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2
            else:
                # 40%: FOREGROUND-CENTERED
                idx = np.random.randint(len(fg_coords[0]))
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
                jitter = int(crop_size * 0.10)
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)
                y1 = cy - crop_size // 2
                x1 = cx - crop_size // 2

            y1 = max(0, min(y1, h - crop_size))
            x1 = max(0, min(x1, w - crop_size))
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)
            image_crop = image[y1:y2, x1:x2]
            mask_crop = mask_2d[y1:y2, x1:x2]
            if image_crop.size == 0 or mask_crop.size == 0:
                continue
            if self._has_sufficient_tissue(image_crop, min_tissue_ratio):
                return image_crop, mask_crop
            white_threshold = 220
            white_mask = np.all(image_crop > white_threshold, axis=2)
            tissue_ratio = 1.0 - (white_mask.sum() / white_mask.size)
            if tissue_ratio > best_tissue_ratio:
                best_tissue_ratio = tissue_ratio
                best_crop = (y1, x1, y2, x2)

        if best_crop is not None:
            y1, x1, y2, x2 = best_crop
            return image[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]
        return image.copy(), mask_2d.copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]

        image = np.array(Image.open(row['image_path']).convert('RGB'))
        mask = np.array(Image.open(row['label_path']).convert('RGB'))

        name = row['name']
        task_id = int(row['task_id'])
        scale_id = int(row['scale_id'])

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        h, w = image.shape[:2]
        crop_size = 512

        # Get crop parameters for this class (SAME as training)
        crop_params = self._get_crop_params(task_id)

        # Apply class-specific cropping (IDENTICAL to training)
        if h > crop_size or w > crop_size:
            if crop_params['strategy'] == 'glomerular_aware':
                # CAP: Glomerular-aware with more centered crops
                image, mask = self._glomerular_aware_crop(image, mask, crop_size, jitter=0.15)
            elif crop_params['strategy'] == 'object_aware':
                # TUFT: Object-aware with mixed sampling
                image, mask = self._object_aware_crop(image, mask, crop_size, jitter=0.3)
            elif crop_params['strategy'] == 'tubule_aware':
                # Tubule-aware with mixed sampling (not used currently)
                image, mask = self._tubule_aware_crop(image, mask, crop_size)
            elif crop_params['strategy'] == 'sparse_aware':
                # PT: Sparse-aware (70% random, 30% FG-aware)
                image, mask = self._sparse_aware_crop(image, mask, crop_size)
            elif crop_params['strategy'] == 'dt_aware':
                # DT: 30% centered, 30% boundary, 40% random
                image, mask = self._dt_aware_crop(image, mask, crop_size)
            elif crop_params['strategy'] == 'vessel_aware':
                # VES: 30% FG-centered, 30% axis-boundary, 40% random
                image, mask = self._vessel_aware_crop(image, mask, crop_size)
            elif crop_params['strategy'] == 'ptc_aware':
                # PTC: 40% FG-centered, 20% boundary, 40% random
                image, mask = self._ptc_aware_crop(image, mask, crop_size)
            else:
                # Fallback: Smart random with foreground constraints
                image, mask = self._smart_random_crop(
                    image, mask, crop_size,
                    min_fg=crop_params['min_fg'],
                    max_fg=crop_params['max_fg'],
                    max_attempts=50
                )

        elif h < crop_size or w < crop_size:
            # Pad small images
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            if HAS_IMGAUG:
                image, mask = self.pad(
                    images=image, 
                    heatmaps=np.expand_dims(mask.astype(np.float32) / 255.0, -1)
                )
                mask = (mask[:, :, :, 0] * 255).astype(np.uint8)

            image = image[0]
            mask = mask[0]

        # Ensure final size matches target
        if image.shape[0] != self.target_h or image.shape[1] != self.target_w:
            image = cv2.resize(image, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)

        mask = (mask >= 127).astype(np.float32)

        image = image.transpose(2, 0, 1).astype(np.float32)
        if image.max() > 1:
            image = image / 255.0

        weight = np.ones_like(mask)

        return {
            'image': image,
            'label': mask,
            'weight': weight,
            'name': name,
            'task_id': task_id,
            'scale_id': scale_id
        }


class FullImageValDataset(Dataset):
    """
    Validation dataset that returns FULL images for sliding window inference.
    
    Unlike PathologyValDataset which crops patches, this returns the complete
    image so that sliding window inference can be performed without using 
    mask information for patch selection.
    
    This enables fair validation where no ground truth information is used
    for determining crop locations.
    """
    
    # Same CLASS_INFO as PathologyDataset for reference
    CLASS_INFO = PathologyDataset.CLASS_INFO
    
    def __init__(
        self,
        csv_path: str,
        max_size: int = 2048,  # Max dimension to avoid OOM
        exclude_classes: Optional[List[int]] = None
    ):
        """
        Args:
            csv_path: Path to validation CSV
            max_size: Maximum image dimension (resize if larger)
            exclude_classes: List of task_ids to exclude
        """
        super().__init__()
        self.csv_path = csv_path
        self.max_size = max_size
        self.exclude_classes = exclude_classes or []
        
        # Load CSV
        self.df = self._load_csv(csv_path)
        
        # Filter excluded classes
        if self.exclude_classes:
            original_len = len(self.df)
            self.df = self.df[~self.df['task_id'].isin(self.exclude_classes)].reset_index(drop=True)
            print(f"FullImageValDataset: Excluded classes {self.exclude_classes}, {original_len} -> {len(self.df)}")
        
        print(f"FullImageValDataset: Loaded {len(self.df)} samples for sliding window validation")
    
    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with auto-detection of separator"""
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                required_cols = ['image_path', 'label_path', 'task_id', 'scale_id']
                if all(col in df.columns for col in required_cols):
                    return df
            except Exception:
                continue
        
        df = pd.read_csv(csv_path, sep=None, engine='python')
        
        column_mapping = {
            'img_path': 'image_path', 'image': 'image_path',
            'mask_path': 'label_path', 'mask': 'label_path', 'label': 'label_path',
            'class_id': 'task_id', 'class': 'task_id', 'task': 'task_id',
            'scale': 'scale_id', 'mag': 'scale_id',
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        return df
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int) -> Dict:
        """Return full image and mask for sliding window inference."""
        row = self.df.iloc[index]
        
        # Load full image and mask
        image = np.array(Image.open(row['image_path']).convert('RGB'))
        mask = np.array(Image.open(row['label_path']).convert('L'))
        
        name = row['name']
        task_id = int(row['task_id'])
        scale_id = int(row['scale_id'])
        
        original_h, original_w = image.shape[:2]
        
        # Resize if too large (to avoid OOM)
        if max(original_h, original_w) > self.max_size:
            scale = self.max_size / max(original_h, original_w)
            new_h, new_w = int(original_h * scale), int(original_w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask
        mask = (mask >= 127).astype(np.float32)
        
        # Normalize image and convert to CHW
        image = image.transpose(2, 0, 1).astype(np.float32)
        if image.max() > 1:
            image = image / 255.0
        
        return {
            'image': image,
            'label': mask,
            'name': name,
            'task_id': task_id,
            'scale_id': scale_id,
            'original_size': (original_h, original_w)
        }


class ClassBalancedSampler(Sampler):
    """
    Sampler that balances classes during training.

    Oversamples minority classes (PTC, VES) to prevent model
    from ignoring rare structures.
    
    NOTE: This shuffles all classes together, so batches may contain mixed classes.
    Use GroupedBatchSampler if you need same-class batches.
    """

    def __init__(
        self,
        dataset: PathologyDataset,
        num_samples_per_class: Optional[Dict[int, int]] = None
    ):
        self.dataset = dataset
        self.df = dataset.df

        # Get class indices
        self.class_indices = {}
        for idx, row in self.df.iterrows():
            task_id = int(row['task_id'])
            if task_id not in self.class_indices:
                self.class_indices[task_id] = []
            self.class_indices[task_id].append(idx)

        # Compute sampling weights
        if num_samples_per_class is None:
            # Default: balance to maximum class
            max_samples = max(len(indices) for indices in self.class_indices.values())
            self.samples_per_class = {k: max_samples for k in self.class_indices.keys()}
        else:
            self.samples_per_class = num_samples_per_class

        self.total_samples = sum(self.samples_per_class.values())

    def __iter__(self):
        indices = []
        for task_id, class_indices in self.class_indices.items():
            n_samples = self.samples_per_class.get(task_id, len(class_indices))
            sampled = np.random.choice(
                class_indices,
                size=n_samples,
                replace=len(class_indices) < n_samples
            )
            indices.extend(sampled)

        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.total_samples


class GroupedBatchSampler(Sampler):
    """
    Sampler that ensures all samples in a batch belong to the SAME class.
    
    This is required for PrPSeg-style architectures where task_id[0] is used
    to select the class token for the entire batch.
    
    Features:
    - Groups samples by task_id
    - Yields batches where all samples have the same class
    - Class-balanced: alternates between classes to ensure equal representation
    - Oversamples minority classes to match majority class
    """

    def __init__(
        self,
        dataset: PathologyDataset,
        batch_size: int,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.df = dataset.df

        # Group indices by class
        self.class_indices = {}
        for idx, row in self.df.iterrows():
            task_id = int(row['task_id'])
            if task_id not in self.class_indices:
                self.class_indices[task_id] = []
            self.class_indices[task_id].append(idx)
        
        self.num_classes = len(self.class_indices)
        self.class_ids = sorted(self.class_indices.keys())
        
        # Compute batches per class (balance to max class)
        max_samples = max(len(indices) for indices in self.class_indices.values())
        self.batches_per_class = (max_samples + batch_size - 1) // batch_size
        
        # Total batches = batches_per_class * num_classes
        self.total_batches = self.batches_per_class * self.num_classes
        
        print(f"GroupedBatchSampler: {self.num_classes} classes, "
              f"{self.batches_per_class} batches/class, "
              f"{self.total_batches} total batches")

    def __iter__(self):
        # Create shuffled indices for each class with oversampling
        class_batches = {}
        for task_id in self.class_ids:
            indices = self.class_indices[task_id].copy()
            
            # Oversample to have enough samples for batches_per_class batches
            target_samples = self.batches_per_class * self.batch_size
            if len(indices) < target_samples:
                # Oversample with replacement
                indices = list(np.random.choice(
                    indices, size=target_samples, replace=True
                ))
            else:
                # Shuffle
                np.random.shuffle(indices)
            
            # Create batches for this class
            batches = []
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
            class_batches[task_id] = batches
        
        # Interleave batches from different classes
        all_batches = []
        for batch_idx in range(self.batches_per_class):
            for task_id in self.class_ids:
                if batch_idx < len(class_batches[task_id]):
                    all_batches.append(class_batches[task_id][batch_idx])
        
        # Shuffle all batches (maintains same-class within batch)
        np.random.shuffle(all_batches)
        
        # Yield individual indices (DataLoader will batch them)
        for batch in all_batches:
            yield from batch

    def __len__(self) -> int:
        return self.total_batches * self.batch_size


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.

    Stacks all tensors and handles metadata.
    """
    images = np.stack([b['image'] for b in batch], axis=0)
    labels = np.stack([b['label'] for b in batch], axis=0)
    weights = np.stack([b['weight'] for b in batch], axis=0)
    names = [b['name'] for b in batch]
    task_ids = np.array([b['task_id'] for b in batch])
    scale_ids = np.array([b['scale_id'] for b in batch])

    return {
        'image': torch.from_numpy(images),
        'label': torch.from_numpy(labels),
        'weight': torch.from_numpy(weights),
        'name': names,
        'task_id': torch.from_numpy(task_ids),
        'scale_id': torch.from_numpy(scale_ids)
    }


# Backward compatibility alias
PathologyValDatasetCustom = PathologyValDataset


if __name__ == '__main__':
    # Test dataset
    print("Testing PathologyDataset...")
    print("\nClass-specific cropping strategies:")
    for task_id, info in PathologyDataset.CLASS_INFO.items():
        print(f"  {info['name']:5s}: strategy={info.get('crop_strategy', 'random'):12s}, "
              f"min_fg={info.get('min_fg', 0.01):.0%}, max_fg={info.get('max_fg', 0.80):.0%}")
