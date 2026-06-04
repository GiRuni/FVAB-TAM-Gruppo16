import cv2
import numpy as np
import sys
import os
import regex as re

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


def parse_target_ids_file(file_path):
    """
    Parse target_img_ids.txt and return a dict:
    {
        "000000301867": {
            "1": [281970],
            "2": [1739135, 1753039, 1762234],
            "3": [1739135, 1753039, 1762234, 281970]
        },
        ...
    }
    """
    relations = {}
    current_image_id = None
    current_relation_num = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Check if it's an image ID line (e.g., "1. 000000301867")
            image_match = re.match(r'^\d+\.\s+(\d{12})', line)
            if image_match:
                current_image_id = image_match.group(1)
                relations[current_image_id] = {}
                continue
            
            # Check if it's a relation line (e.g., "   1. umbrella + white (281970)")
            relation_match = re.match(r'^\s+(\d+)\.\s+(.+)', line)
            if relation_match and current_image_id:
                current_relation_num = relation_match.group(1)
                relation_text = relation_match.group(2)
                
                # Extract all numbers (IDs) from the relation text
                ids = re.findall(r'\d+', relation_text)
                # Convert to integers and remove duplicates while preserving order
                unique_ids = []
                seen = set()
                for id_str in ids:
                    id_int = int(id_str)
                    if id_int not in seen:
                        unique_ids.append(id_int)
                        seen.add(id_int)
                
                relations[current_image_id][current_relation_num] = unique_ids

                if int(current_relation_num) in (1, 2):
                    # Extract attributes and actions subjects and related masks

                    attr_action_pattern = re.compile(
                        r"^\s*(?P<subject>[\w\s]+?)\s*\+\s*"                    # Subject name
                        r"(?P<aux>[\w\s]+?)\s*"                                 # Attribute or Action name
                        r"\((?P<subject_masks>[\d+,\s]+)"                       # Subject masks
                        r"(?:\s*,\s*\((?P<action_masks>[\d+,\s]+)\))?\)\s*$"    # Optional nested action masks (da non includere nel merge)
                    )

                    attr_action_matches = re.match(attr_action_pattern, relation_text)

                    subject_name = attr_action_matches.group("subject")
                    subject_ids = [x.strip() for x in attr_action_matches.group("subject_masks").split(',')]

                    relations[current_image_id][f"{current_relation_num}_{subject_name}"] = subject_ids
                else:
                    # Extract spatial subjects and related masks

                    spatial_pattern = re.compile(
                        r"^\s*(?P<first_subject>[\w\s]+?)\s*\+\s*"              # First subject name
                        r"(?P<spatial>[\w\s]+?)\s*\+\s*"                        # Spatial keyword
                        r"(?P<second_subject>[\w\s]+?)\s*"                      # Second subject name
                        r"\(\s*"                                                # Opening outer parenthesis
                        r"(?:\((?P<masks_a>[\d+,\s]+)\)|(?P<masks_a>\d+))"      # First subject: either (id, id) or a single id
                        r"\s*,\s*"                                              # Separating comma
                        r"(?:\((?P<masks_b>[\d+,\s]+)\)|(?P<masks_b>\d+))"      # Second subject: either (id, id) or a single id
                        r"\s*\)$"                                               # Closing outer parenthesis
                    )

                    spatial_matches = re.match(spatial_pattern, relation_text)

                    first_subject_name = spatial_matches.group("first_subject")
                    first_subject_ids = [x.strip() for x in spatial_matches.group("masks_a").split(',')]

                    second_subject_name = spatial_matches.group("second_subject")
                    second_subject_ids = [x.strip() for x in spatial_matches.group("masks_b").split(',')]

                    relations[current_image_id][f"{current_relation_num}_{first_subject_name}_first"] = first_subject_ids
                    relations[current_image_id][f"{current_relation_num}_{second_subject_name}_second"] = second_subject_ids

    return relations


if __name__ == "__main__":
    # Static paths
    root_dir = r"/content/FVAB-TAM-Gruppo16/tam-logit-lenses/ll_tam/masks"
    target_file = r"/content/FVAB-TAM-Gruppo16/Fase_3/target_img_ids.txt"
    output_dir = r"/content/FVAB-TAM-Gruppo16/Fase_3/merged_masks"

    # Parse the target file
    try:
        relations = parse_target_ids_file(target_file)
    except Exception as e:
        print(f"[ERROR] Failed to parse target file: {e}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Process each image and its relations
    for image_id, image_relations in relations.items():
        image_folder = os.path.join(root_dir, image_id)

        # Check if image folder exists
        if not os.path.isdir(image_folder):
            print(f"[SKIP] {image_id}: folder not found")
            continue

        # Create output folder for this image
        image_output_dir = os.path.join(output_dir, image_id)
        os.makedirs(image_output_dir, exist_ok=True)

        # Process each relation
        for relation_num, mask_ids in image_relations.items():
            if not mask_ids:  # Skip empty relations
                continue

            try:
                # Find mask files corresponding to the IDs
                mask_files = []
                for mask_id in mask_ids:
                    # Try different extensions
                    found = False
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                        potential_file = os.path.join(image_folder, f"{mask_id}{ext}")
                        if os.path.exists(potential_file):
                            mask_files.append(potential_file)
                            found = True
                            break
                    if not found:
                        print(f"[WARN] {image_id} relation {relation_num}: mask {mask_id} not found")

                if not mask_files:
                    print(f"[SKIP] {image_id} relation {relation_num}: no masks found")
                    continue

                # Combine masks
                if len(mask_files) == 1:
                    # Single mask → just load & binarize
                    mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"[ERROR] {image_id} relation {relation_num}: failed to load mask {mask_files[0]}")
                        continue
                    _, combined_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                else:
                    # Multiple masks → combine all
                    combined_mask = combine_multiple_masks(mask_files)

                output_path = os.path.join(image_output_dir, f"{relation_num}.png")
                success = cv2.imwrite(output_path, combined_mask)
                
                if success:
                    print(f"[OK] {image_id} relation {relation_num} ({len(mask_files)} masks) -> {output_path}")
                else:
                    print(f"[ERROR] {image_id} relation {relation_num}: failed to write output to {output_path}")

            except Exception as e:
                print(f"[ERROR] {image_id} relation {relation_num}: {e}")
