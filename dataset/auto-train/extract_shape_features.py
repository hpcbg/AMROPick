import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
import time

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. Configuration ---
INPUT_IMAGE_FOLDER = "datas/yolo_datasets/yolo_metal_dataset_sam2_ori_plus_2025-06-17-simeon/images/train"
OUTPUT_CSV_FILE = "datas/datasets/shape_features/shape_features-test.csv"
MASKS_OUTPUT_FOLDER = "datas/datasets/shape_features/masks"
SAM_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MAX_ANGLES_TO_STORE = 10

# --- 2. Setup Models and Directories ---
print("🛠️ Loading SAM 2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(MODEL_CFG, SAM_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print("✅ Model loaded.")

# Create the output directory for masks if it doesn't exist
os.makedirs(MASKS_OUTPUT_FOLDER, exist_ok=True)


# --- 3. Feature Extraction & Helper Functions ---
def get_angles(poly):
    """Calculates the internal angles of a polygon's vertices."""
    angles = []
    points = poly.reshape(-1, 2)
    num_points = len(points)
    if num_points < 3: return []

    for i in range(num_points):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[(i + 1) % num_points]
        
        vec_a, vec_b = p_prev - p_curr, p_next - p_curr
        dot_product = np.dot(vec_a, vec_b)
        mag_a, mag_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
        
        if mag_a * mag_b == 0: continue
            
        angle_rad = np.arccos(np.clip(dot_product / (mag_a * mag_b), -1.0, 1.0))
        angles.append(np.degrees(angle_rad))
        
    return angles

def clean_mask(mask):
    """Keeps only the largest connected component of a binary mask."""
    num_labels, labels_matrix, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels_matrix == largest_label)
    return mask

def extract_features(mask):
    """Calculates an extended set of geometric features from a binary mask."""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 20: return None

    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    
    # --- Minimum Area Rectangle Features ---
    min_area_rect = cv2.minAreaRect(contour)
    min_rect_w, min_rect_h = min_area_rect[1]
    rotation_angle = min_area_rect[2]
    min_rect_area = min_rect_w * min_rect_h
    rectangularity = area / min_rect_area if min_rect_area > 0 else 0

    # --- Internal Angles ---
    epsilon = 0.02 * perimeter
    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
    angles = get_angles(approx_poly)
    # Normalize angles: sort and pad to a fixed length
    sorted_angles = sorted(angles, reverse=True)
    normalized_angles = np.zeros(MAX_ANGLES_TO_STORE)
    num_angles = min(len(sorted_angles), MAX_ANGLES_TO_STORE)
    normalized_angles[:num_angles] = sorted_angles[:num_angles]
    
    # --- Existing Features ---
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    num_vertices = len(approx_poly)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i])) if hu_moments[i] != 0 else 0
    num_holes = sum(1 for h in hierarchy[0] if h[3] != -1)

    # --- Assemble Dictionary ---
    feature_dict = {
        'area': area, 'perimeter': perimeter, 'aspect_ratio': aspect_ratio,
        'extent': extent, 'solidity': solidity, 'circularity': circularity,
        'min_rect_w': min_rect_w, 'min_rect_h': min_rect_h, 
        'rotation_angle': rotation_angle, 'rectangularity': rectangularity,
        'num_vertices': num_vertices, 'num_holes': num_holes
    }
    for i, hu in enumerate(hu_moments):
        feature_dict[f'hu_moment_{i+1}'] = hu
    for i, angle in enumerate(normalized_angles):
        feature_dict[f'angle_{i+1}'] = angle
        
    return feature_dict

# --- 4. Main Application ---
class FeatureExtractorTool:
    def __init__(self, class_label):
        self.class_label = class_label
        
        self.image_files = sorted([f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.current_img_idx = 0
        self.new_mask_proposal = None
        self.quit = False
        self.fig, self.ax = plt.subplots(figsize=(15, 15))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    # on_click, update_display, change_image, etc. are the same as before
    def load_current_image(self):
        image_name = self.image_files[self.current_img_idx]
        image_path = os.path.join(INPUT_IMAGE_FOLDER, image_name)
        self.current_image = cv2.imread(image_path)
        print(f"\nLoading image {self.current_img_idx + 1}/{len(self.image_files)}. Setting predictor...")
        predictor.set_image(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        self.update_display()

    def update_display(self):
        image_name = self.image_files[self.current_img_idx]
        title = ""
        display_image = self.current_image.copy()
        if self.new_mask_proposal is not None:
            title = "Accept this mask? (y)es / (n)o"
            overlay = display_image.copy()
            contours, _ = cv2.findContours(self.new_mask_proposal.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.fillPoly(overlay, contours, color=(0, 255, 0))
            display_image = cv2.addWeighted(overlay, 0.45, display_image, 0.55, 0)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        else:
            title = (f"Image {self.current_img_idx + 1}/{len(self.image_files)}: {image_name}\n"
                     f"Annotating as '{self.class_label}' | LEFT-CLICK to segment | (d) Next | (a) Previous | (q) Quit")
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        self.ax.set_title(title)
        self.ax.axis('off')
        plt.draw()


    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1 or self.new_mask_proposal is not None: return
        point_coords = np.array([[int(event.xdata), int(event.ydata)]])
        point_labels = np.array([1])
        print("Predicting new mask...")
        masks, scores, _ = predictor.predict(point_coords, point_labels, multimask_output=True)
        raw_mask = masks[np.argmax(scores)]
        self.new_mask_proposal = clean_mask(raw_mask)
        self.update_display()


    def on_key_press(self, event):
        if self.new_mask_proposal is not None:
            if event.key == 'y':
                print("Extracting and saving features...")
                
                # 1. Save the mask array to a .npy file
                mask_filename = f"{int(time.time()*1000)}_{self.class_label}.npy"
                mask_path = os.path.join(MASKS_OUTPUT_FOLDER, mask_filename)
                np.save(mask_path, self.new_mask_proposal)
                
                # 2. Extract features from the mask
                features = extract_features(self.new_mask_proposal)
                
                if features:
                    features['class_label'] = self.class_label
                    features['source_image'] = self.image_files[self.current_img_idx]
                    features['mask_file'] = mask_filename # Add reference to the mask file
                    self.save_to_csv(features)

                self.new_mask_proposal = None
            elif event.key == 'n':
                self.new_mask_proposal = None
            self.update_display()
        
        else: # Browse mode
            if event.key in ['d', 'right']: self.change_image(1)
            elif event.key in ['a', 'left']: self.change_image(-1)
            elif event.key == 'q': self.quit = True; plt.close(self.fig)

    def change_image(self, delta):
        self.current_img_idx = (self.current_img_idx + delta + len(self.image_files)) % len(self.image_files)
        self.load_current_image()

    def save_to_csv(self, feature_dict):
        file_exists = os.path.isfile(OUTPUT_CSV_FILE)
        # Define fieldnames based on the first dictionary to ensure order
        fieldnames = list(feature_dict.keys())
        with open(OUTPUT_CSV_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(feature_dict)
        print(f"Features and mask reference saved to {OUTPUT_CSV_FILE}")

    def run(self):
        self.load_current_image()
        plt.show(block=True)

if __name__ == "__main__":
    class_labels = input("Enter the class names, separated by commas (e.g., partA,partB,partC): ").strip().split(',')
    print("\nPlease select which class you will be annotating:")
    for i, label in enumerate(class_labels):
        print(f"  [{i}] {label.strip()}")
    
    try:
        choice_idx = int(input(f"Enter the number (0-{len(class_labels)-1}): "))
        if not 0 <= choice_idx < len(class_labels): raise ValueError
        chosen_label = class_labels[choice_idx].strip()
        print(f"\nStarting session. All accepted masks will be labeled as '{chosen_label}'.")
        
        tool = FeatureExtractorTool(chosen_label)
        tool.run()
        print("\nFeature extraction session finished.")

    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")