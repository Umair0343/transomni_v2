import os
import re
import argparse

# Mapping of task types to IDs and scale IDs
TASK_INFO = {
    'dt':  {'task_id': 0, 'scale_id': 1},
    'pt':  {'task_id': 1, 'scale_id': 1},
    'cap': {'task_id': 2, 'scale_id': 0},
    'tuft':{'task_id': 3, 'scale_id': 0},
    'art': {'task_id': 4, 'scale_id': 1},   # art/vessel combined
    'ptc': {'task_id': 5, 'scale_id': 3}
}

def generate_tsv(root_dir, output_tsv):
    entries = []

    # Traverse the dataset folder
    for layer_name in sorted(os.listdir(root_dir)):
        layer_path = os.path.join(root_dir, layer_name)
        if not os.path.isdir(layer_path):
            continue

        # Extract task from folder name
        # Example: 1_0_1_dt -> dt
        task_key = layer_name.split('_')[-1].lower()
        if task_key not in TASK_INFO:
            print(f"Skipping unknown task: {layer_name}")
            continue
        task_id = TASK_INFO[task_key]['task_id']
        scale_id = TASK_INFO[task_key]['scale_id']

        for stain_name in sorted(os.listdir(layer_path)):
            stain_path = os.path.join(layer_path, stain_name)
            if not os.path.isdir(stain_path):
                continue

            # Find all images (exclude masks)
            image_files = [f for f in os.listdir(stain_path) if not re.search(r'_mask', f)]
            for img_file in sorted(image_files):
                img_path = os.path.join(stain_path, img_file)

                # Extract numeric ID
                m = re.match(r'im_(\d+)\.', img_file)
                if not m:
                    print(f"Skipping file with unexpected name: {img_file}")
                    continue
                img_id = m.group(1)

                # Find corresponding mask
                mask_candidates = [f for f in os.listdir(stain_path)
                                   if re.search(rf'im_{img_id}_mask.*\.(png|tif)$', f)]
                if not mask_candidates:
                    print(f"No mask found for {img_file} in {stain_path}")
                    continue
                mask_file = mask_candidates[0]
                mask_path = os.path.join(stain_path, mask_file)

                # Compose name field
                name = f"{layer_name}_{stain_name}_im_{img_id}.png"

                # Layer ID from first number in folder name
                layer_id = int(layer_name.split('_')[0])

                entries.append([
                    img_path,
                    mask_path,
                    name,
                    str(layer_id),
                    str(task_id),
                    str(scale_id)
                ])

    # Write TSV (tab-separated)
    with open(output_tsv, 'w') as f:
        f.write("image_path\tlabel_path\tname\tlayer_id\ttask_id\tscale_id\n")
        for row in entries:
            f.write("\t".join(row) + "\n")

    print(f"TSV generated with {len(entries)} entries at {output_tsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TSV for HATs dataset")
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to the dataset root (e.g., train folder)')
    parser.add_argument('--output_tsv', type=str, required=True,
                        help='Path to output TSV file')
    args = parser.parse_args()

    generate_tsv(args.root_dir, args.output_tsv)
