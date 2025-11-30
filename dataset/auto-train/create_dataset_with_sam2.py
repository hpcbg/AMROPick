import os
import cv2
import torch
import numpy as np
import random
import yaml
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- 1. Configuration ---
INPUT_FOLDER = "/home/carlos/Documents/S1S2/Simeon/datas/2025-06-17-simeon"
OUTPUT_DATASET_FOLDER = "yolo_metal_dataset_sam2_2025-06-17-simeon_test"
CLASS_NAME = "metal part"
TRAIN_VAL_SPLIT = 0.8

SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# --- 2. Setup SAM 2 Model ---
print("🛠️ Loading SAM 2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

sam2 = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
print("✅ SAM 2 Model loaded.")

# --- 3. Find Images and Create Dataset Structure ---
all_image_paths = []
supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
for root, _, files in os.walk(INPUT_FOLDER):
    for file in files:
        if file.lower().endswith(supported_extensions):
            all_image_paths.append(os.path.join(root, file))

random.shuffle(all_image_paths)
split_index = int(len(all_image_paths) * TRAIN_VAL_SPLIT)
train_images = all_image_paths[:split_index]
val_images = all_image_paths[split_index:]

print(f"Found {len(all_image_paths)} total images.")
print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")

subsets = {"train": train_images, "val": val_images}
for subset in subsets.keys():
    os.makedirs(os.path.join(OUTPUT_DATASET_FOLDER, "images", subset), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATASET_FOLDER, "labels", subset), exist_ok=True)

# --- 4. Process Images and Generate Annotations ---
def process_subset(image_paths, subset_name):
    """Processes a list of images and saves them in YOLO format."""
    print(f"\nProcessing {subset_name} set...")
    for image_path in tqdm(image_paths, desc=f"Creating {subset_name} data"):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read {image_path}. Skipping.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape

            # Generate masks using SAM 2 model
            masks = mask_generator.generate(image_rgb)

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(OUTPUT_DATASET_FOLDER, "labels", subset_name, f"{base_filename}.txt")
            
            with open(label_path, "w") as f:
                for ann in masks:
                    segmentation = ann['segmentation']
                    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    class_id = 0
                    
                    for contour in contours:
                        if contour.size < 6:
                            continue
                        
                        norm_contour = contour.flatten().astype(float)
                        norm_contour[0::2] /= img_w
                        norm_contour[1::2] /= img_h
                        
                        polygon_str = " ".join(map(str, norm_contour))
                        f.write(f"{class_id} {polygon_str}\n")
            
            output_image_path = os.path.join(OUTPUT_DATASET_FOLDER, "images", subset_name, os.path.basename(image_path))
            cv2.imwrite(output_image_path, image)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

for subset_name, image_list in subsets.items():
    process_subset(image_list, subset_name)

# --- 5. Create data.yaml file ---
yaml_data = {
    'train': f'../{OUTPUT_DATASET_FOLDER}/images/train',
    'val': f'../{OUTPUT_DATASET_FOLDER}/images/val',
    'nc': 1,
    'names': [CLASS_NAME]
}

yaml_path = os.path.join(OUTPUT_DATASET_FOLDER, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"\n🎉 YOLO dataset created successfully at: {OUTPUT_DATASET_FOLDER}")
print(f"YAML configuration file saved at: {yaml_path}")