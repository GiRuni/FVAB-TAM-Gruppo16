import cv2
import numpy as np
import sys
import os

def combine_masks(mask_path1, mask_path2, output_path):
    # Load masks as grayscale
    mask1 = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)

    if mask1 is None or mask2 is None:
        raise ValueError("Error loading one of the mask images.")

    # Ensure same size
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same dimensions.")

    # Convert to binary (optional, if masks are not already 0/255)
    _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

    # Sum masks (clip to 255 to avoid overflow)
    combined = np.clip(mask1.astype(np.uint16) + mask2.astype(np.uint16), 0, 255).astype(np.uint8)

    # Save result
    cv2.imwrite(output_path, combined)
    print(f"Combined mask saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_masks.py mask1.png mask2.png output.png")
        sys.exit(1)

    mask1_path = sys.argv[1]
    mask2_path = sys.argv[2]
    output_path = sys.argv[3]

    combine_masks(mask1_path, mask2_path, output_path)