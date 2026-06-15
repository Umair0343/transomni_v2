"""
generate_masks.py
=================
Generate binary prediction masks for all 6 tissue classes using the trained
HybridScaleTransformer model (--backbone custom).

USAGE
-----
1. Fill in the 6 class path lists below (dt, pt, cap, tuft, art, ptc).
   Each variable is a Python list of absolute image file paths.

2. Set CHECKPOINT_PATH to your saved .pth model file.

3. Set OUTPUT_DIR to a folder where predicted masks will be saved.
   Sub-folders are created automatically:
       <OUTPUT_DIR>/DT/
       <OUTPUT_DIR>/PT/
       <OUTPUT_DIR>/CAP/
       <OUTPUT_DIR>/TUFT/
       <OUTPUT_DIR>/ART/
       <OUTPUT_DIR>/PTC/

4. Run:
       python generate_masks.py

Notes
-----
- Works on images of any size (512×512, 1024×1024, 3000×3000, etc.).
- For images larger than 512×512 a sliding-window approach extracts
  non-overlapping 512×512 patches, runs inference on each, and stitches
  the probability maps back before thresholding at 0.5.
- Output masks are saved as 8-bit PNG files (0 = background, 255 = foreground).
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────────────────
#  USER CONFIGURATION  ← edit everything in this block
# ─────────────────────────────────────────────────────────

# Path to your trained model checkpoint (.pth file)
CHECKPOINT_PATH = "/media/iml1/umair/TransNetR/code1/hybrid_cnn_transformer/output/custom_v7_6class/best_model.pth"

# Output folder – sub-folders per class are created automatically
OUTPUT_DIR = "/media/iml1/umair/TransNetR/code1/predicted_masks2"

# Inference settings
PATCH_SIZE   = 512          # Do NOT change – model was trained on 512×512 patches
OVERLAP      = 0.5         # Overlap between patches during sliding-window inference
THRESHOLD    = 0.5          # Probability threshold for binary mask
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Image paths per class ──────────────────────────────────
# Replace the placeholder strings with your actual image file paths.
# You can add as many paths as you want to each list.

dt = [
    #"/media/iml1/umair/TransNetR/code1/validation/dt/im_38_patch_1024_0.png",
    # "/path/to/dt/image2.tif",
]

pt = [
    #"/media/iml1/umair/TransNetR/code1/validation/pt/im_48_patch_0_1976.png",
]

cap = [
    #"/media/iml1/umair/TransNetR/code1/validation/cap/im_78.png",
]

tuft = [
    #"/media/iml1/umair/TransNetR/code1/validation/tuft/im_188.png",
]

art = [
    "/media/iml1/umair/TransNetR/code1/validation/art/im_24_patch_1977_1024.png",
]

ptc = [
    #"/media/iml1/umair/TransNetR/code1/validation/ptc/pas_ptc_KI_PTC_PAS_MCD_im_1_p512_1536_a2fe2db0b895456b903f49212fcb6f66.png",
]

# ─────────────────────────────────────────────────────────
#  END OF USER CONFIGURATION
# ─────────────────────────────────────────────────────────

# Internal class registry
# task_id matches the order used during training (train.py: ALL_CLASS_INFO)
#   0 → DT   (scale_id = 1, 10×)
#   1 → PT   (scale_id = 1, 10×)
#   2 → CAP  (scale_id = 0,  5×)
#   3 → TUFT (scale_id = 0,  5×)
#   4 → VES  (scale_id = 1, 10×) ← arteries
#   5 → PTC  (scale_id = 3, 40×)
CLASS_REGISTRY = [
    {"name": "DT",   "task_id": 0, "scale_id": 1, "paths_var": "dt"},
    {"name": "PT",   "task_id": 1, "scale_id": 1, "paths_var": "pt"},
    {"name": "CAP",  "task_id": 2, "scale_id": 0, "paths_var": "cap"},
    {"name": "TUFT", "task_id": 3, "scale_id": 0, "paths_var": "tuft"},
    {"name": "ART",  "task_id": 4, "scale_id": 1, "paths_var": "art"},
    {"name": "PTC",  "task_id": 5, "scale_id": 3, "paths_var": "ptc"},
]

# Map variable names to the actual lists defined above
PATH_VARS = {
    "dt":   dt,
    "pt":   pt,
    "cap":  cap,
    "tuft": tuft,
    "art":  art,
    "ptc":  ptc,
}


# ─── Model loading ────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained HybridScaleTransformer from a checkpoint."""
    # Make sure the project root is on the Python path so imports work
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import after path is set
    from models import build_hybrid_model  # noqa: F401

    print(f"Building model (backbone=custom, num_task_classes=6) …")
    model = build_hybrid_model(
        num_task_classes=6,
        num_scales=4,
        use_transformer=True,
        backbone="custom",
        pretrained_backbone=False,  # weights come from checkpoint
        dropout=0.1,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Support different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint  # assume raw state dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    model = model.to(device)
    model.eval()
    print(f"Model loaded and moved to {device}.\n")
    return model


# ─── Image loading ────────────────────────────────────────

def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Load an image file and return a float32 tensor in [0, 1] with shape [3, H, W].
    Grayscale images are converted to RGB by repeating the channel.
    """
    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    tensor = TF.to_tensor(img)  # [3 or 1, H, W], values in [0, 1]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]

    return tensor.float()  # [3, H, W]


# ─── Hanning window weight ────────────────────────────────

def _make_hanning_window(size: int, device: torch.device) -> torch.Tensor:
    """
    Build a 2-D Hanning (raised-cosine) weight window of shape [size, size].
    Values range from ~0 at the edges to 1.0 at the centre.
    This tapers patch contributions to zero at boundaries so that overlapping
    patches blend smoothly with no visible seam lines.
    """
    h1d = torch.hann_window(size, periodic=False, device=device)   # [size]
    h2d = torch.outer(h1d, h1d)                                     # [size, size]
    # Avoid exact zeros at borders (they would kill edge pixels even with overlap)
    h2d = h2d.clamp(min=1e-6)
    return h2d


# ─── Sliding-window inference ─────────────────────────────

@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    task_id: int,
    scale_id: int,
    patch_size: int = 512,
    overlap: float = 0.50,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Run Hanning-weighted sliding-window inference and return a full-resolution
    probability map (float32 tensor, shape [H, W], values in [0, 1]).

    Hanning weighting eliminates patch-boundary seam artefacts:
    each patch is multiplied by a raised-cosine window before accumulation,
    so the centre of every patch contributes fully while edges fade to zero.
    Overlapping patches therefore blend continuously across the whole image.

    For images that are ≤ 512×512 the image is processed directly
    without any tiling.
    """
    if device is None:
        device = next(model.parameters()).device

    image = image.to(device)
    C, H, W = image.shape

    task_tensor  = torch.tensor([task_id],  dtype=torch.long, device=device)
    scale_tensor = torch.tensor([scale_id], dtype=torch.long, device=device)

    # ── Direct inference for small images (≤ patch_size) ──────────────
    if H <= patch_size and W <= patch_size:
        pad_h = patch_size - H
        pad_w = patch_size - W
        img_padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h))
        outputs = model(img_padded, task_tensor, scale_tensor)
        prob = torch.sigmoid(outputs["seg_logits"][:, 1])   # [1, ph, pw]
        prob = prob[:, :H, :W].squeeze(0)                    # crop back → [H, W]
        return prob.cpu()

    # ── Hanning-weighted sliding window for larger images ──────────────
    stride = max(1, int(patch_size * (1.0 - overlap)))

    # Weight window shared across all patches
    win = _make_hanning_window(patch_size, device)   # [P, P]

    weighted_sum = torch.zeros(H, W, dtype=torch.float32, device=device)
    weight_total = torch.zeros(H, W, dtype=torch.float32, device=device)

    def get_starts(total_size: int):
        """Return all patch start positions, always including the far edge."""
        starts = list(range(0, total_size - patch_size + 1, stride))
        last = total_size - patch_size
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    y_starts = get_starts(H) if H >= patch_size else [0]
    x_starts = get_starts(W) if W >= patch_size else [0]

    for y in y_starts:
        for x in x_starts:
            patch = image[:, y:y + patch_size, x:x + patch_size].unsqueeze(0)  # [1,C,P,P]
            outputs     = model(patch, task_tensor, scale_tensor)
            prob_patch  = torch.sigmoid(outputs["seg_logits"][:, 1]).squeeze(0)  # [P,P]

            # Weight the probability map by the Hanning window
            weighted_sum[y:y + patch_size, x:x + patch_size] += prob_patch * win
            weight_total[y:y + patch_size, x:x + patch_size] += win

    weight_total = weight_total.clamp(min=1e-6)
    return (weighted_sum / weight_total).cpu()


# ─── Save mask ────────────────────────────────────────────

def save_mask(prob_map: torch.Tensor, output_path: str, threshold: float = 0.5):
    """Threshold probability map and save as 8-bit PNG (0 / 255)."""
    binary = (prob_map >= threshold).numpy().astype(np.uint8) * 255
    mask_img = Image.fromarray(binary, mode="L")
    mask_img.save(output_path)


# ─── Main ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Prediction Mask Generator")
    print("  HybridScaleTransformer  |  backbone=custom  |  6 classes")
    print("=" * 60)

    # Validate checkpoint
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Please set CHECKPOINT_PATH in generate_masks.py"
        )

    device = torch.device(DEVICE)
    model  = load_model(CHECKPOINT_PATH, device)

    # Create output sub-folders
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls_info in CLASS_REGISTRY:
        os.makedirs(os.path.join(OUTPUT_DIR, cls_info["name"]), exist_ok=True)

    total_processed = 0
    total_skipped   = 0

    for cls_info in CLASS_REGISTRY:
        cls_name  = cls_info["name"]
        task_id   = cls_info["task_id"]
        scale_id  = cls_info["scale_id"]
        image_paths = PATH_VARS[cls_info["paths_var"]]

        if not image_paths:
            print(f"[{cls_name}] No images provided – skipping.")
            continue

        print(f"\n[{cls_name}]  task_id={task_id}  scale_id={scale_id}  "
              f"({len(image_paths)} image(s))")

        out_dir = os.path.join(OUTPUT_DIR, cls_name)

        for img_path in image_paths:
            if not os.path.isfile(img_path):
                print(f"  ✗ File not found: {img_path}")
                total_skipped += 1
                continue

            # Derive output filename
            base_name   = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(out_dir, f"{base_name}_predicted_mask.png")

            print(f"  Processing: {os.path.basename(img_path)}", end=" … ", flush=True)

            try:
                image = load_image_as_tensor(img_path)  # [3, H, W]
                _, H, W = image.shape
                print(f"({H}×{W})", end=" ", flush=True)

                prob_map = sliding_window_inference(
                    model, image, task_id, scale_id,
                    patch_size=PATCH_SIZE, overlap=OVERLAP, device=device,
                )

                save_mask(prob_map, output_path, threshold=THRESHOLD)
                print(f"→ saved to {output_path}")
                total_processed += 1

            except Exception as e:
                print(f"\n  ✗ ERROR processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                total_skipped += 1

    print("\n" + "=" * 60)
    print(f"Done!  Processed: {total_processed}  Skipped/Failed: {total_skipped}")
    print(f"Masks saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
