"""
Run inference with the best trained model on a few validation images.

For each selected validation sample this script saves, in a folder named after
the image, the original image, ground-truth mask, predicted mask, a colour-coded
overlay (TP/FP/FN), and a metrics.txt with the per-image Dice/precision/recall.

It is robust to the checkpoint format:
  - accepts checkpoints saved with key 'model' OR 'model_state_dict'
  - infers num_task_classes from the saved cls_emb tensor
  - loads with strict=False and reports any mismatched keys

Image normalisation matches training exactly (image / 255.0), NOT ImageNet stats.

Usage:
    python run_inference.py \
        --checkpoint scalehybrid_v2/runs/best.pth \
        --val_csv val_list.csv \
        --num_images 2 \
        --output_dir inference_results
"""

import os
import sys
import argparse
import csv as _csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Make the local `models` package importable regardless of folder name
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.hybrid_scale_transformer import HybridScaleTransformer  # noqa: E402

CLASS_NAMES = {0: 'DT', 1: 'PT', 2: 'CAP', 3: 'TUFT', 4: 'VES', 5: 'PTC'}


def read_csv_rows(csv_path):
    """Read the tab-separated dataset CSV into a list of dict rows."""
    rows = []
    with open(csv_path, 'r') as f:
        # The dataset CSVs are tab-separated
        reader = _csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)
    return rows


def load_model(checkpoint_path, device):
    """Load model, inferring num_task_classes from the checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Locate the state dict (support both save formats)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt  # raw state dict

    # Report any extra info stored in the checkpoint
    if isinstance(ckpt, dict):
        if 'epoch' in ckpt:
            print(f"  checkpoint epoch: {ckpt['epoch']}")
        if 'best_dice' in ckpt:
            print(f"  checkpoint best_dice: {ckpt['best_dice']}")
        if 'per_class' in ckpt:
            print(f"  checkpoint per_class: {ckpt['per_class']}")

    # Infer number of task classes from the saved class-embedding table
    num_task_classes = 6
    for key in ('cnn_encoder.cls_emb', 'cls_emb'):
        if key in state:
            num_task_classes = state[key].shape[1]
            break
    print(f"  inferred num_task_classes: {num_task_classes}")

    model = HybridScaleTransformer(
        num_classes=2,
        num_task_classes=num_task_classes,
        num_scales=4,
        backbone='custom',
        use_transformer=True,
        dropout=0.1,
    )

    # Drop any keys whose shapes do not match the current model. load_state_dict
    # raises on a size mismatch even with strict=False, so we filter first. A
    # shape mismatch usually means the checkpoint came from a different
    # architecture version (e.g. trained WITH the stem that has since been removed).
    model_state = model.state_dict()
    filtered, shape_mismatch = {}, []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        elif k in model_state:
            shape_mismatch.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    total = len(model_state)
    print(f"  loaded {loaded}/{total} parameter tensors")
    if shape_mismatch:
        print(f"  WARNING: {len(shape_mismatch)} tensors had a SHAPE mismatch and were skipped.")
        print(f"           first few: {shape_mismatch[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} checkpoint keys not in current model (skipped).")
    if shape_mismatch or (loaded < total * 0.9):
        print("  *** ARCHITECTURE MISMATCH: this checkpoint does not fully match the")
        print("      current model code. Predictions will be UNRELIABLE. You likely need")
        print("      the code version that trained it, or to retrain on this architecture. ***")

    model = model.to(device)
    model.eval()
    return model, num_task_classes


@torch.no_grad()
def predict(model, image_rgb, task_id, scale_id, device, patch_size=512):
    """
    Run inference on a single image (sliding window if larger than patch_size).
    image_rgb: [H, W, 3] uint8.  Returns [H, W] binary uint8 prediction.
    """
    h, w = image_rgb.shape[:2]

    def run_patch(patch):
        # Match training normalisation exactly: divide by 255, CHW
        t = torch.from_numpy(patch.transpose(2, 0, 1).astype(np.float32) / 255.0)
        t = t.unsqueeze(0).to(device)
        tid = torch.tensor([task_id], dtype=torch.long, device=device)
        sid = torch.tensor([scale_id], dtype=torch.long, device=device)
        out = model(t, tid, sid)
        prob = torch.sigmoid(out['seg_logits'][:, 1])  # [1, H, W]
        return prob.squeeze(0).cpu().numpy()

    if h <= patch_size and w <= patch_size:
        pad_h, pad_w = patch_size - h, patch_size - w
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            padded = image_rgb
        prob = run_patch(padded)[:h, :w]
        return (prob > 0.5).astype(np.uint8)

    # Sliding window for larger images
    stride = patch_size // 2
    prob_map = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    ys = list(range(0, max(1, h - patch_size + 1), stride)) or [0]
    xs = list(range(0, max(1, w - patch_size + 1), stride)) or [0]
    if ys[-1] != h - patch_size and h > patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size and w > patch_size:
        xs.append(w - patch_size)
    for y in ys:
        for x in xs:
            patch = image_rgb[y:y + patch_size, x:x + patch_size]
            prob_map[y:y + patch_size, x:x + patch_size] += run_patch(patch)
            count[y:y + patch_size, x:x + patch_size] += 1
    prob_map /= np.maximum(count, 1)
    return (prob_map > 0.5).astype(np.uint8)


def make_overlay(image_rgb, gt_binary, pred_binary, alpha=0.55):
    """Colour-coded overlay: blue=TP, yellow=FP, magenta=FN."""
    tp = (pred_binary == 1) & (gt_binary == 1)
    fp = (pred_binary == 1) & (gt_binary == 0)
    fn = (pred_binary == 0) & (gt_binary == 1)
    out = image_rgb.astype(np.float32).copy()
    out[tp] = out[tp] * (1 - alpha) + np.array([0, 100, 255]) * alpha
    out[fp] = out[fp] * (1 - alpha) + np.array([255, 255, 0]) * alpha
    out[fn] = out[fn] * (1 - alpha) + np.array([255, 0, 255]) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), tp.sum(), fp.sum(), fn.sum()


def main():
    parser = argparse.ArgumentParser(description='Run inference on validation images')
    parser.add_argument('--checkpoint', type=str, default='scalehybrid_v2/runs/best.pth')
    parser.add_argument('--val_csv', type=str, default='val_list.csv')
    parser.add_argument('--num_images', type=int, default=2,
                        help='Number of validation images to run on')
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--patch_size', type=int, default=512)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    model, _ = load_model(args.checkpoint, device)

    rows = read_csv_rows(args.val_csv)
    if not rows:
        print(f"No rows found in {args.val_csv}")
        return
    rows = rows[:args.num_images]
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, row in enumerate(rows):
        img_path = row['image_path']
        mask_path = row['label_path']
        name = row.get('name') or os.path.basename(img_path)
        task_id = int(row['task_id'])
        scale_id = int(row['scale_id'])
        cls_name = CLASS_NAMES.get(task_id, str(task_id))

        print(f"\n[{idx + 1}/{len(rows)}] {name}  (task={cls_name}, scale={scale_id})")

        if not os.path.exists(img_path):
            print(f"  SKIP: image not found: {img_path}")
            continue

        image = np.array(Image.open(img_path).convert('RGB'))
        gt = np.zeros(image.shape[:2], dtype=np.uint8)
        if os.path.exists(mask_path):
            gt = (np.array(Image.open(mask_path).convert('L')) > 127).astype(np.uint8)
        else:
            print(f"  WARNING: mask not found, ground-truth metrics will be 0: {mask_path}")

        pred = predict(model, image, task_id, scale_id, device, patch_size=args.patch_size)

        overlay, tp, fp, fn = make_overlay(image, gt, pred)
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Folder named after the image (without extension)
        folder_name = os.path.splitext(name)[0]
        out_dir = os.path.join(args.output_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        Image.fromarray(image).save(os.path.join(out_dir, 'image.png'))
        Image.fromarray((gt * 255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_mask.png'))
        Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(out_dir, 'pred_mask.png'))
        Image.fromarray(overlay).save(os.path.join(out_dir, 'overlay.png'))

        with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
            f.write(f"name:      {name}\n")
            f.write(f"class:     {cls_name} (task_id={task_id})\n")
            f.write(f"scale_id:  {scale_id}\n")
            f.write(f"image:     {img_path}\n")
            f.write(f"mask:      {mask_path}\n")
            f.write(f"TP: {int(tp)}  FP: {int(fp)}  FN: {int(fn)}\n")
            f.write(f"Dice:      {dice:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")

        print(f"  Dice={dice:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
        print(f"  saved -> {out_dir}/")

    print(f"\nDone. Results in: {args.output_dir}/")


if __name__ == '__main__':
    main()
