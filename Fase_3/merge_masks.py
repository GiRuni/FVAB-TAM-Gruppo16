import cv2
import numpy as np
import sys
import os

def combine_multiple_masks(mask_paths):
    masks = []

    for path in mask_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Error loading mask: {path}")

        # Binarize
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masks.append(mask)

    # Check dimensions
    shapes = [m.shape for m in masks]
    if len(set(shapes)) != 1:
        raise ValueError("Masks must have the same dimensions.")

    # Combine all masks
    combined = np.zeros_like(masks[0], dtype=np.uint16)
    for m in masks:
        combined += m.astype(np.uint16)

    combined = np.clip(combined, 0, 255).astype(np.uint8)
    return combined


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python merge_masks.py <root_masks_folder>")
        sys.exit(1)

    root_dir = sys.argv[1]
    output_dir = os.path.join(root_dir, "merged_seg_label")

    os.makedirs(output_dir, exist_ok=True)

    for image_id in os.listdir(root_dir):
        image_folder = os.path.join(root_dir, image_id)

        # Skip output folder
        if image_id == "merged_seg_label":
            continue

        if not os.path.isdir(image_folder):
            continue

        # Collect mask files
        mask_files = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        if len(mask_files) == 0:
            print(f"Skipping {image_id}: no masks found")
            continue

        try:
            if len(mask_files) == 1:
                # Single mask → just load & binarize
                mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
                _, combined_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            else:
                # Multiple masks → combine all
                combined_mask = combine_multiple_masks(mask_files)

            output_path = os.path.join(output_dir, f"{image_id}.png")
            cv2.imwrite(output_path, combined_mask)

            print(f"[OK] {image_id} ({len(mask_files)} masks) -> {output_path}")

        except Exception as e:
            print(f"[ERROR] {image_id}: {e}")
