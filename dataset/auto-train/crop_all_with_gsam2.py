import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

import os
import sys
sys.path.append('gsam2/autodistill-grounded-sam-2')

import torch
import numpy as np
import cv2

# Import what you need, note that matplotlib is commented out as we won't show plots in batch mode
# import matplotlib.pyplot as plt 

from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

# --- Model and Ontology Setup ---
ontology = CaptionOntology(
    {
        "metal part": "metal part",
        "aluminium part": "aluminium part", 
        "metal block": "metal block",
        "aluminium block": "aluminium block",
        "single metal part on a pile of metal parts": "metal part",
        "colored metal part": "metal part",
        "colored aluminium part": "aluminium part",
    }
)

print("🛠️ Loading GroundedSAM2 model...")
base_model = GroundedSAM2(ontology=ontology, model="Grounding DINO")
print("✅ Model loaded.")

# --- Define Input and Output Folders ---
input_folder = "/home/carlos/Documents/S1S2/Simeon/datas"
output_folder = "cropped_output"
os.makedirs(output_folder, exist_ok=True)

# --- Find All Image Files Recursively ---
image_paths = []
supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(supported_extensions):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images to process.")

# --- Loop Through Each Image and Process It ---
for image_path in image_paths:
    print(f"\n--- Processing: {image_path} ---")
    try:
        # Run prediction on the current image
        results = base_model.predict(image_path)
        
        # Check if any masks were detected
        if hasattr(results, 'mask') and len(results.mask) > 0:
            # Load the original image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image file: {image_path}")
                continue
                
            h, w, _ = image.shape

            # Combine all masks into one single mask
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in results.mask:
                uint8_mask = mask.astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, uint8_mask)

            # Find the bounding box of the combined mask
            points = cv2.findNonZero(combined_mask)
            
            if points is not None:
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(points)

                # Crop the original image
                cropped_image = image[y:y+h, x:x+w]

                # Save the cropped image
                base_filename = os.path.basename(image_path)
                output_filename = f"cropped_{base_filename}"
                output_path = os.path.join(output_folder, output_filename)

                cv2.imwrite(output_path, cropped_image)
                print(f"✅ Saved cropped image to: {output_path}")
            else:
                print("Could not find bounding box for the combined mask.")
        else:
            print("No masks were detected.")

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")

print("\n🎉 Batch processing complete.")