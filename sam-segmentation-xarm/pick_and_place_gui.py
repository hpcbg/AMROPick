import cv2
import numpy as np
import time
import threading
import os
import json

import utils.json_config
from segmentation.segment import SegmentWithSAM
from segmentation.detect import ObjectDetector


class PickAndPlaceApp:
    def __init__(self):
        self.config = utils.json_config.load(
            "./config/pick_and_place_gui.json")
        self.cap = None if self.config['static_image_mode'] else cv2.VideoCapture(
            self.config['camera_URL'])

        self.current_frame = None
        self.display_frame = None
        self.processing = False
        self.selected_object = None

        self.window_width = self.config['image_width']
        self.window_height = self.config['image_height'] + \
            self.config['button_bar_height']
        self.button_height = self.config['button_bar_height']

        self.buttons = self.config['buttons']

        cv2.namedWindow(self.config['window_name'])
        cv2.setMouseCallback(self.config['window_name'], self.mouse_callback)

        self.segment = SegmentWithSAM(self.config['SAM_checkpoint'])
        self.detect = ObjectDetector()
        self.objects = utils.json_config.load(
            "./config/objects.json")
        print(self.objects)

    def mouse_callback(self, event, x, y, flags, param):
        global search_ratio, search_area_ratio, search_area
        if event == cv2.EVENT_LBUTTONDOWN and not self.processing:
            # Check if button is pressed
            x = x / self.config['scale_factor']
            y = y / self.config['scale_factor']
            for btn in self.buttons:
                if (btn["x"] <= x <= btn["x"] + btn["width"] and
                        btn["y"] <= y <= btn["y"] + btn["height"]):
                    self.selected_object = btn["name"]
                    print(f"Selected object: {self.selected_object}")

                    threading.Thread(target=self.process_frame,
                                     daemon=True).start()
                    break

    def draw_buttons(self, frame):
        for btn in self.buttons:
            # Inactive button
            color = btn["color"] if not self.processing else (100, 100, 100)
            cv2.rectangle(frame, (btn["x"], btn["y"]),
                          (btn["x"] + btn["width"], btn["y"] + btn["height"]),
                          color, -1)
            cv2.rectangle(frame, (btn["x"], btn["y"]),
                          (btn["x"] + btn["width"], btn["y"] + btn["height"]),
                          (255, 255, 255), 2)

            # Button text
            text_size = cv2.getTextSize(
                btn["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = btn["x"] + (btn["width"] - text_size[0]) // 2
            text_y = btn["y"] + (btn["height"] + text_size[1]) // 2
            cv2.putText(frame, btn["name"], (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def add_text_overlay(self, frame, text, color=(0, 255, 0)):
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 20, text_y - text_size[1] - 20),
                      (text_x + text_size[0] + 20, text_y + 20), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        return frame

    def step1_segment(self, frame):
        self.masks = self.segment.get_masks(frame)
        return self.segment.plot_masks(frame, self.masks)

    def step2_detect(self, frame):
        return self.detect.detect_object(frame,
                                         self.masks,
                                         self.objects[self.selected_object],
                                         self.config['workarea']['fence'])

    def step3_send_to_robot(self, pick_pose):
        path = "robot_status.json"
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r") as f:
                self.robot_status = json.load(f)
        except:
            return False

        if self.robot_status.get("status", "UNKNOWN") == "OK":
            data = {
                "pick_pose": {
                    "x": pick_pose['x'] + self.objects[self.selected_object]['offset']['x'],
                    "y": pick_pose['y'] + self.objects[self.selected_object]['offset']['y'],
                    "z": -25,
                    "roll_degrees": 0,
                    "pitch_degrees": 180,
                    "yaw_degrees": 0  # pick_pose['yaw']
                },
                "place_pose": {
                    "x": self.config['place_pose']['x'],
                    "y": self.config['place_pose']['y'],
                    "z": self.config['place_pose']['z'],
                    "roll_degrees": 0,
                    "pitch_degrees": 180,
                    "yaw_degrees": 0,
                }
            }

            with open("pick_and_place.json", "w") as f:
                json.dump(data, f, indent=4)

            print("Created pick_and_place.json:", data)
            return True

        return False

    def process_frame(self):
        self.processing = True

        if self.cap:
            cap = cv2.VideoCapture(self.config['camera_URL'])
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.processing = False
                return
        else:
            frame = cv2.imread(self.config['static_image'])

        original_frame = frame.copy()

        self.display_frame = original_frame.copy()

        # Step 1: Segmentation
        self.display_frame = self.add_text_overlay(
            original_frame.copy(), "Running SAM segmentation...", (0, 255, 255))

        step1_result = self.step1_segment(original_frame)
        self.display_frame = step1_result.copy()
        cv2.putText(self.display_frame, "Result from segmentation", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Step 2: Detection
        self.display_frame = self.add_text_overlay(
            self.display_frame.copy(), "Searching for object...", (0, 255, 255))

        step2_result, pick_pose = self.step2_detect(original_frame)
        self.display_frame = step2_result.copy()
        cv2.putText(self.display_frame, "Result from object detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if pick_pose is not None:
            pick_pose = self.detect.convert_to_global_coordinates(
                pick_pose, self.config['workarea'])
            # Step 3: Pick and place
            self.display_frame = self.add_text_overlay(
                step2_result.copy(), "Sending coordinates to robot...", (0, 255, 255))
            time.sleep(1.5)

            step3_result = self.step3_send_to_robot(pick_pose)
            self.display_frame = step2_result.copy()
            cv2.putText(self.display_frame, "Coordinates sent to robot." if step3_result else "Robot was not available.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(self.display_frame, f"x: {int(pick_pose['x'])}, y: {int(pick_pose['y'])}, yaw: {int(pick_pose['yaw'])}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            time.sleep(5)
        else:
            self.display_frame = self.add_text_overlay(
                step2_result.copy(), "Object is not detected.", (0, 255, 255))
        time.sleep(5)

        self.processing = False
        self.display_frame = None
        print(f"Completed for {self.selected_object}")

    def get_robot_status(self):
        path = "robot_status.json"
        if not os.path.exists(path):
            self.robot_status = {"status": "UNKNOWN", "pos": None}

        try:
            with open(path, "r") as f:
                self.robot_status = json.load(f)
        except:
            self.robot_status = {"status": "ERROR", "pos": None}

    def draw_robot_status(self, canvas):
        self.get_robot_status()

        status = self.robot_status.get("status", "UNKNOWN")

        cv2.putText(canvas, "ROBOT IS OK" if status == "OK" else "ROBOT IS BUSY", (2100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def run(self):
        while True:
            if self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    break
            else:
                frame = cv2.imread(self.config['static_image'])

            if self.display_frame is not None:
                display = self.display_frame.copy()
            else:
                display = frame.copy()
                if not self.processing:
                    cv2.putText(display, "Choose object", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            canvas = np.zeros(
                (self.window_height, self.window_width, 3), dtype=np.uint8)
            canvas[self.button_height:, :] = display

            self.draw_buttons(canvas)

            self.draw_robot_status(canvas)

            cv2.imshow(self.config['window_name'], cv2.resize(
                canvas, None, fx=self.config['scale_factor'], fy=self.config['scale_factor']))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = PickAndPlaceApp()
    app.run()
