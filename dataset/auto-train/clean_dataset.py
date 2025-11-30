import os
import cv2
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. Configuration ---
DATASET_ROOT_FOLDER = "datas/yolo_datasets/yolo_metal_dataset_sam2_2025-06-17-simeon_test"
SAM_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CLASS_ID = 0

# --- 2. Setup SAM2 Model and Predictor ---
print("🛠️ Loading SAM 2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(MODEL_CFG, SAM_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print("✅ Model loaded.")

# --- 3. Helper Functions ---
def parse_yolo_label(label_path, img_w, img_h):
    if not os.path.exists(label_path): return []
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            polygon_norm = np.array(parts[1:], dtype=np.float32)
            annotations.append({
                "class_id": int(parts[0]),
                "polygon_norm": polygon_norm,
                # Assign a random color to each existing mask for clear visualization
                "color": tuple(np.random.randint(50, 255, 3).tolist())
            })
    return annotations

def save_yolo_label(label_path, annotations):
    with open(label_path, 'w') as f:
        for ann in annotations:
            polygon_str = " ".join(map(str, ann['polygon_norm']))
            f.write(f"{ann['class_id']} {polygon_str}\n")

def mask_to_yolo_polygon(mask, img_w, img_h):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contour = max(contours, key=cv2.contourArea)
    if contour.size < 6: return None
    norm_contour = contour.flatten().astype(float)
    norm_contour[0::2] /= img_w
    norm_contour[1::2] /= img_h
    return norm_contour

def redraw_display(ax, image, annotations, new_mask_proposal=None):
    ax.clear()
    display_image = image.copy()
    img_h, img_w, _ = display_image.shape

    # Draw existing masks with their assigned random colors
    for ann in annotations:
        poly_denorm = ann['polygon_norm'].copy()
        poly_denorm[0::2] *= img_w
        poly_denorm[1::2] *= img_h
        overlay = display_image.copy()
        cv2.fillPoly(overlay, [poly_denorm.astype(np.int32).reshape(-1, 2)], color=ann['color'])
        display_image = cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0)

    # Draw the newly generated mask in bright green for review
    if new_mask_proposal is not None:
        overlay = display_image.copy()
        contours, _ = cv2.findContours(new_mask_proposal.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(overlay, contours, color=(0, 255, 0)) # Bright green
        display_image = cv2.addWeighted(overlay, 0.5, display_image, 0.5, 0)
    
    ax.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    plt.draw()

# --- 4. Main Application Logic ---
class AnnotationTool:
    def __init__(self, subset):
        self.subset = subset
        self.image_folder = os.path.join(DATASET_ROOT_FOLDER, "images", subset)
        self.label_folder = os.path.join(DATASET_ROOT_FOLDER, "labels", subset)
        self.image_files = sorted([f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.current_img_idx = 0
        self.annotations = []
        self.new_mask_proposal = None
        self.mode = "Browse" # "Browse" or "reviewing"
        self.quit = False

        self.fig, self.ax = plt.subplots(figsize=(15, 15))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def load_current_image(self):
        image_name = self.image_files[self.current_img_idx]
        image_path = os.path.join(self.image_folder, image_name)
        self.current_image = cv2.imread(image_path)
        h, w, _ = self.current_image.shape
        
        label_path = os.path.join(self.label_folder, os.path.splitext(image_name)[0] + ".txt")
        self.annotations = parse_yolo_label(label_path, w, h)
        
        print(f"Loading image {self.current_img_idx + 1}/{len(self.image_files)}. Setting predictor...")
        predictor.set_image(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        self.update_display()

    def update_display(self):
        image_name = self.image_files[self.current_img_idx]
        if self.mode == "Browse":
            title = (f"Image {self.current_img_idx + 1}/{len(self.image_files)}: {image_name}\n"
                     f"L-CLICK: Add Mask | R-CLICK: Delete Mask | (d) Next | (a) Previous | (q) Quit")
        elif self.mode == "reviewing":
            title = "Accept this new mask? (y)es / (n)o"
        
        redraw_display(self.ax, self.current_image, self.annotations, self.new_mask_proposal)
        self.ax.set_title(title, fontsize=12)
        plt.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or self.mode != "Browse": return
        
        click_point = (int(event.xdata), int(event.ydata))
        
        # LEFT-CLICK: Add new mask
        if event.button == 1:
            point_coords = np.array([click_point])
            point_labels = np.array([1])
            
            print("Predicting new mask...")
            masks, scores, _ = predictor.predict(point_coords, point_labels, multimask_output=True)
            
            self.new_mask_proposal = masks[np.argmax(scores)]
            self.mode = "reviewing"
            self.update_display()
        
        # RIGHT-CLICK: Delete existing mask
        elif event.button == 3:
            h, w, _ = self.current_image.shape
            mask_to_delete = -1
            # Iterate backwards to find the top-most mask under the click
            for i in reversed(range(len(self.annotations))):
                ann = self.annotations[i]
                poly_denorm = ann['polygon_norm'].copy()
                poly_denorm[0::2] *= w
                poly_denorm[1::2] *= h
                contour = poly_denorm.astype(np.int32).reshape(-1, 2)
                
                # Check if the click was inside this polygon
                if cv2.pointPolygonTest(contour, click_point, False) >= 0:
                    mask_to_delete = i
                    break
            
            if mask_to_delete != -1:
                print(f"Deleting mask {mask_to_delete + 1}")
                self.annotations.pop(mask_to_delete)
                self.update_display()


    def on_key_press(self, event):
        if self.mode == "reviewing":
            if event.key == 'y': # Accept new mask
                h, w, _ = self.current_image.shape
                new_polygon = mask_to_yolo_polygon(self.new_mask_proposal, w, h)
                if new_polygon is not None:
                    self.annotations.append({
                        "class_id": CLASS_ID, 
                        "polygon_norm": new_polygon,
                        "color": tuple(np.random.randint(50, 255, 3).tolist())
                    })
                self.new_mask_proposal = None
                self.mode = "Browse"
            elif event.key == 'n': # Reject new mask
                self.new_mask_proposal = None
                self.mode = "Browse"
            
        elif self.mode == "Browse":
            if event.key in ['d', 'right']: self.save_and_change_image(1)
            elif event.key in ['a', 'left']: self.save_and_change_image(-1)
            elif event.key == 'q': self.save_and_change_image(0, quit_app=True)
        
        self.update_display()

    def save_and_change_image(self, delta, quit_app=False):
        image_name = self.image_files[self.current_img_idx]
        label_path = os.path.join(self.label_folder, os.path.splitext(image_name)[0] + ".txt")
        save_yolo_label(label_path, self.annotations)
        print(f"Saved {len(self.annotations)} masks for {image_name}")

        if quit_app:
            self.quit = True
            plt.close(self.fig)
            return

        self.current_img_idx = (self.current_img_idx + delta + len(self.image_files)) % len(self.image_files)
        self.load_current_image()

    def run(self):
        self.load_current_image()
        plt.show(block=True)

if __name__ == "__main__":
    subset_choice = input("Which subset do you want to edit? (train/val): ").strip().lower()
    if subset_choice in ["train", "val"]:
        tool = AnnotationTool(subset_choice)
        tool.run()
    print("\nEditing session finished.")