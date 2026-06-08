"""
label_generate.py
-----------------
Recursively generates labels.txt for the IDNet dataset.

Each line in the output file contains:
  <relative_path_from_DATASET_ROOT_DIR> <label_id>

Example output:
  arizona_dl/positive/img_001.tif 0
  california_dl/fraud2_face_morphing/forged_010.tif 2
  washingtondc_dl/fraud3_face_replacement/fake_021.tif 3

This script supports multi-state folders and 5 fraud/genuine classes.
If classes needs to be changed, edit class_to_label for corresponding classes
"""

import os
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_ROOT_DIR = "/Users/vtech/Desktop/f25-16/fraud-detection/data/IDNET/ID Card"
OUTPUT_FILE = "/Users/vtech/Desktop/f25-16/fraud-detection/labels/IDC_labels.txt"
# ---------------------

# Define your class-to-label mapping
class_to_label = {
    'positive': 0,
    'fraud1_copy_and_move': 1,
    'fraud2_face_morphing': 2,
    'fraud3_face_replacement': 3,
    'fraud4_combined': 4
}


def generate_labels_file_recursively():
    print(f"Scanning dataset directory: {DATASET_ROOT_DIR}")
    if not os.path.isdir(DATASET_ROOT_DIR):
        print(f" Error: Directory not found at {DATASET_ROOT_DIR}")
        return

    all_labels = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    # Traverse all state folders and their subdirectories
    for root, dirs, files in tqdm(os.walk(DATASET_ROOT_DIR), desc="Scanning Folders"):
        class_name = os.path.basename(root)
        if class_name in class_to_label:
            label = class_to_label[class_name]
            for image_filename in files:
                if image_filename.lower().endswith(image_extensions):
                    # Create relative path from DATASET_ROOT_DIR to the image
                    rel_path = os.path.relpath(os.path.join(root, image_filename), DATASET_ROOT_DIR)
                    all_labels.append(f"{rel_path} {label}")

    if not all_labels:
        print(" No images found. Please check your dataset path and folder structure.")
        return

    # Sort the entries for consistent ordering
    all_labels.sort()

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(all_labels))

    print(f"\n Successfully generated labels file at: {OUTPUT_FILE}")
    print(f" Total images indexed: {len(all_labels)}")
    print(f" Example entry: {all_labels[0]}")


if __name__ == "__main__":
    generate_labels_file_recursively()

