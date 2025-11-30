import os
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. Configuration ---
DATASET_ROOT_FOLDER = "datas/yolo_datasets/yolo_metal_dataset_sam2_2025-06-17-simeon"
# Remove masks with a pixel area smaller than this value.
MIN_AREA_THRESHOLD = 250
# Remove masks with circularity greater than or equal to this value (1.0 is a perfect circle).
CIRCULARITY_THRESHOLD = 0.75

# --- 2. Helper Function ---
def calculate_circularity(contour):
    """Calculates the circularity of a contour."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Avoid division by zero for invalid contours
    if perimeter == 0:
        return 0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

# --- 3. Main Cleaning Logic ---
def clean_dataset_subset(subset_name):
    """Processes a subset (e.g., 'train' or 'val') of the dataset."""
    print(f"\n--- Cleaning '{subset_name}' subset ---")
    
    image_folder = os.path.join(DATASET_ROOT_FOLDER, "images", subset_name)
    label_folder = os.path.join(DATASET_ROOT_FOLDER, "labels", subset_name)

    if not os.path.exists(label_folder):
        print(f"Warning: Label folder not found at {label_folder}. Skipping.")
        return 0, 0

    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    
    total_masks_removed = 0
    total_files_removed = 0

    for label_name in tqdm(label_files, desc=f"Processing {subset_name} labels"):
        label_path = os.path.join(label_folder, label_name)
        image_name = os.path.splitext(label_name)[0] + ".jpg" # Assumes .jpg, adjust if needed
        image_path = os.path.join(image_folder, image_name)
        
        # Check for corresponding image with different extensions if .jpg not found
        if not os.path.exists(image_path):
            found_image = False
            for ext in ['.png', '.jpeg', '.bmp', '.webp']:
                temp_path = os.path.join(image_folder, os.path.splitext(label_name)[0] + ext)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    found_image = True
                    break
            if not found_image:
                print(f"Warning: No corresponding image found for {label_name}. Skipping.")
                continue

        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        kept_annotations = []
        original_mask_count = len(lines)

        for line in lines:
            parts = line.strip().split()
            polygon_norm = np.array(parts[1:], dtype=np.float32)

            # De-normalize coordinates to calculate real pixel area and perimeter
            poly_denorm = polygon_norm.copy()
            poly_denorm[0::2] *= img_w
            poly_denorm[1::2] *= img_h
            contour = poly_denorm.astype(np.int32).reshape(-1, 2)
            
            # Check the geometric properties
            area = cv2.contourArea(contour)
            circularity = calculate_circularity(contour)

            # Keep the mask only if it passes both checks
            if area >= MIN_AREA_THRESHOLD and circularity < CIRCULARITY_THRESHOLD:
                kept_annotations.append(line)
        
        masks_removed_in_file = original_mask_count - len(kept_annotations)
        total_masks_removed += masks_removed_in_file

        # If all masks were removed, delete the label and image file
        if not kept_annotations:
            os.remove(label_path)
            os.remove(image_path)
            total_files_removed += 1
        # Otherwise, overwrite the label file with the cleaned annotations
        elif masks_removed_in_file > 0:
            with open(label_path, 'w') as f:
                f.writelines(kept_annotations)

    return total_masks_removed, total_files_removed

if __name__ == "__main__":
    grand_total_masks = 0
    grand_total_files = 0
    
    for subset in ["train", "val"]:
        masks_removed, files_removed = clean_dataset_subset(subset)
        grand_total_masks += masks_removed
        grand_total_files += files_removed
    
    print("\n--- Cleaning Complete ---")
    print(f"Total masks removed: {grand_total_masks}")
    print(f"Total image/label pairs removed: {grand_total_files}")