import math
import cv2
from ultralytics import YOLO

from .helpers import resize_to_square, get_precise_box


class Vision:
    def __init__(self,
                 detect_model_path="yolo11n.pt",
                 classify_model_path="yolo11n-cls.pt",
                 class_names=[],
                 z_offsets=[],
                 class_colors=[(255, 0, 0),
                               (0, 0, 255),
                               (0, 255, 255),
                               (255, 0, 255),
                               (255, 255, 0),
                               (255, 255, 255),
                               (0, 0, 0)]):
        self.detect_model = YOLO(detect_model_path)
        self.classify_model = YOLO(classify_model_path)
        self.z_offsets = z_offsets
        self.class_names = class_names
        self.class_colors = class_colors

        self.image = None
        self.processed_image = None
        self.detected_objects = []
        self.best_objects = {}

    def load_image_from_file(self, image_path):
        self.image = cv2.imread(image_path)

    def load_image_from_cv2(self, image):
        self.image = image.copy()

    def process_image(self):
        if self.image is None:
            raise ValueError("Image not loaded. Please load an image first.")

        objects = self.detect_model(
            self.image, conf=0.6, agnostic_nms=True, iou=0.3)

        self.processed_image = self.image.copy()
        self.detected_objects = []
        self.best_objects = {}

        for box in objects[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(self.processed_image, (x1, y1),
                          (x2, y2), (0, 0, 0), 1)

            object_image = resize_to_square(self.image[y1:y2, x1:x2])

            border_w = 15
            precise_box, (x, y, theta) = get_precise_box(
                self.image[y1 - border_w: y2 + border_w, x1 - border_w: x2 + border_w])
            cv2.circle(self.processed_image, (x + x1 - border_w,
                       y + y1 - border_w), 9, (0, 255, 0), -1)
            start = (x + x1 - border_w, y + y1 - border_w)
            end = (int(start[0] - math.sin(math.radians(theta)) * 70),
                   int(start[1] - math.cos(math.radians(theta)) * 70))
            cv2.line(self.processed_image, start, end, (0, 255, 0), 3)

            classify_results = self.classify_model(object_image)

            try:
                if classify_results and hasattr(classify_results[0], "probs"):
                    predicted_class_name = f"{self.class_names[classify_results[0].probs.top1]}"
                else:
                    predicted_class_name = "Unknown"
            except:
                predicted_class_name = "Unknown"

            try:
                color = self.class_colors[classify_results[0].probs.top1]
            except:
                color = (0, 255, 0)

            cv2.drawContours(self.processed_image, [
                             precise_box + [x1 - border_w, y1 - border_w]], -1, color, 3)

            cv2.putText(self.processed_image, predicted_class_name, (x1, y2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            # For the simulation in vision_pick_and_place.wbt:
            #   x -> [129; 1791] =  [-0.4; 0.4]
            #   y -> [159; 1404] = -[-0.3; 0.3]
            detected_object = {
                "name": self.class_names[classify_results[0].probs.top1],
                "confidence": classify_results[0].probs.top1conf.cpu().numpy(),
                "x": 0.8 * (x+x1-border_w - (1791 + 129) / 2) / (1791 - 129),
                "y": - 0.6 * (y+y1-border_w - (1404 + 159) / 2) / (1404 - 159),
                "z": self.z_offsets[classify_results[0].probs.top1],
                "theta": math.radians(theta)
            }

            self.detected_objects.append(detected_object)
            if detected_object["name"] not in self.best_objects or \
                    self.best_objects[detected_object["name"]]["confidence"] < detected_object["confidence"]:
                self.best_objects[detected_object["name"]] = detected_object

    def get_processed_image(self):
        return self.processed_image

    def get_detected_objects(self):
        return self.detected_objects

    def get_object(self, object_name):
        if object_name in self.best_objects:
            return self.best_objects[object_name]
        else:
            return None
