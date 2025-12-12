import cv2
import os
import uuid

import json_config


CONFIG = json_config.load("./config/pick_and_place_gui.json")
OUTPUT_DIR = "captured"

camera_url = CONFIG['camera_URL']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# cap = cv2.VideoCapture(RTSP_URL)

# if not cap.isOpened():
#     print("Error: Cannot open RTSP stream.")
#     exit()

print("Press SPACE to save image, ESC to exit.")

while True:
    cap = cv2.VideoCapture(camera_url)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Stream error.")
        break

    cv2.imshow("Capture", cv2.resize(
                frame, None, fx=CONFIG['scale_factor'], fy=CONFIG['scale_factor']))
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    if key == 32:  # SPACE
        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(path, frame)
        print("Saved:", path)

cap.release()
cv2.destroyAllWindows()
