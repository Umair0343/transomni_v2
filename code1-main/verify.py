import csv

def check_all_masks(csv_file):
    """
    Check image-mask pairs with support for different mask types
    """
    
    print("Checking image-mask pairs...")
    print("=" * 80)
    
    errors = []
    matches = 0
    row_count = 0
    
    with open(csv_file, 'r') as f:
        # Auto-detect delimiter
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        f.seek(0)
        
        reader = csv.reader(f, delimiter=delimiter)
        
        # Skip header row
        headers = next(reader)
        print(f"Headers: {headers}")
        
        for line_num, row in enumerate(reader, 2):  # Start from 2 because we skipped header
            if len(row) < 2:
                continue
                
            row_count += 1
            
            # Get image and mask paths
            image_path = row[0].strip()
            mask_path = row[1].strip()
            
            # Get just the filenames
            image_filename = image_path.split('/')[-1]
            mask_filename = mask_path.split('/')[-1]
            
            # Extract base names
            # For image: remove extension
            if '.' in image_filename:
                img_base = image_filename.rsplit('.', 1)[0]
            else:
                img_base = image_filename
            
            # For mask: remove '_mask_xxx' and extension
            mask_base = mask_filename
            
            # Remove common mask patterns
            mask_patterns = [
                '_mask_distal',
                '_mask_PTC_lumen', 
                '_mask_capsule',
                '_mask_proximal',
                '_mask_'
            ]
            
            for pattern in mask_patterns:
                if pattern in mask_base:
                    mask_base = mask_base.split(pattern)[0]
                    break
            
            # Remove extension from mask base
            if '.' in mask_base:
                mask_base = mask_base.rsplit('.', 1)[0]
            
            # Check if they match
            if img_base == mask_base:
                matches += 1
                if line_num <= 10:  # Show first few matches
                    print(f"Row {line_num}: ✓ {image_filename} ↔ {mask_filename}")
            else:
                errors.append((line_num, image_filename, mask_filename))
                if line_num <= 10:  # Show first few errors
                    print(f"Row {line_num}: ✗ {image_filename} ≠ {mask_filename}")
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY:")
    print(f"Total data rows: {row_count}")
    print(f"Correct matches: {matches}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nFIRST 10 ERRORS:")
        for i, (line_num, img, mask) in enumerate(errors[:10]):
            print(f"Row {line_num}: Image={img}, Mask={mask}")
        
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors")
    
    return errors

def analyze_mask_types(csv_file):
    """
    Analyze what types of masks exist in the CSV
    """
    
    print("\n" + "=" * 80)
    print("ANALYZING MASK TYPES:")
    print("=" * 80)
    
    mask_types = {}
    
    with open(csv_file, 'r') as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        f.seek(0)
        
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 2:
                continue
                
            mask_path = row[1].strip()
            mask_filename = mask_path.split('/')[-1]
            
            # Extract mask type
            if '_mask_' in mask_filename:
                # Extract everything after '_mask_'
                parts = mask_filename.split('_mask_')
                if len(parts) > 1:
                    mask_type = parts[1].split('.')[0]  # Remove extension
                    mask_types[mask_type] = mask_types.get(mask_type, 0) + 1
    
    print("\nMask types found:")
    for mask_type, count in sorted(mask_types.items()):
        print(f"  {mask_type}: {count} masks")
    
    return mask_types

# Main execution
if __name__ == "__main__":
    csv_file = input("Enter path to CSV file: ").strip()
    if not csv_file:
        csv_file = "/media/iml1/umair/TransNetR/code1/hybrid_cnn_transformer/train_list.csv"
    
    print(f"Checking file: {csv_file}")
    print("=" * 80)
    
    # First analyze what mask types we have
    mask_types = analyze_mask_types(csv_file)
    
    # Then check all pairs
    errors = check_all_masks(csv_file)
    
    if not errors:
        print("\n✅ SUCCESS: All image-mask pairs are correct!")
    else:
        print(f"\n⚠️  Found {len(errors)} potential issues")
        
        # Ask user what to do
        action = input("\nDo you want to: \n1. See all errors\n2. Save errors to file\n3. Exit\nChoice (1-3): ").strip()
        
        if action == '1':
            print("\nALL ERRORS:")
            for line_num, img, mask in errors:
                print(f"Row {line_num}: Image={img}, Mask={mask}")
        
        elif action == '2':
            output_file = "mask_errors.csv"
            with open(output_file, 'w') as f:
                f.write("row,image,mask\n")
                for line_num, img, mask in errors:
                    f.write(f"{line_num},{img},{mask}\n")
            print(f"Errors saved to: {output_file}")