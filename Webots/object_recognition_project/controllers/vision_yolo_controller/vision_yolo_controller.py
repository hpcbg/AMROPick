from controller import Robot

from ultralytics import YOLO
import cv2
import numpy as np
import base64
import math


model_detect = YOLO("detect.pt")
model_classify = YOLO("classify.pt")

robot = Robot()

camera = robot.getDevice("camera")
camera.enable(50)

receiver = robot.getDevice("receiver")
receiver.enable(50)

emitter = robot.getDevice("emitter")

timestep = int(robot.getBasicTimeStep())

detected_objects = {}


def resize_to_square(image, size=640):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    square_image = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    square_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return square_image


def get_precise_box(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    main_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        center_x, center_y = 0, 0

    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    angle = rect[2]

    width, height = rect[1]
    if width < height:
        angle = 90 + angle
    angle = 90 - angle

    return main_contour, (center_x, center_y, angle)


def process_image(image):
    global detected_objects
    detected_objects = {}

    results = model_detect(image, conf=0.6, agnostic_nms=True, iou=0.3)

    image_filtered = image.copy()

    class_names = {0: "Plate1", 1: "Plate3"}
    class_colors = {0: (255, 0, 0), 1: (0, 0, 255)}

    res = results[0].boxes

    detected_boxes = []
    for box in res:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = box.conf.cpu().numpy()[0]

        detected_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(image_filtered, (x1, y1), (x2, y2), (0, 0, 0), 1)

        object_image = resize_to_square(image[y1:y2, x1:x2])

        border_w = 15
        precise_box, (x, y, theta) = get_precise_box(
            image[y1-border_w:y2+border_w, x1-border_w:x2+border_w])
        cv2.circle(image_filtered, (x + x1 - border_w, y +
                   y1 - border_w),   9,  (0, 255, 0),  -1)

        start = (x + x1 - border_w, y + y1 - border_w)
        end = (int(start[0] - math.sin(math.radians(theta)) * 90),
               int(start[1] - math.cos(math.radians(theta)) * 90))

        image = cv2.line(image_filtered, start, end, (0, 255, 0), 3)

        classify_results = model_classify(object_image)

        if classify_results and hasattr(classify_results[0], 'probs'):
            # predicted_class_name = f"{class_names[classify_results[0].probs.top1]} ({classify_results[0].probs.top1conf.cpu().numpy():.2f})"
            predicted_class_name = f"{class_names[classify_results[0].probs.top1]}"
        else:
            predicted_class_name = "Unknown"

        color = class_colors.get(classify_results[0].probs.top1, (0, 255, 0))
        cv2.drawContours(image_filtered, [
                         precise_box+[x1-border_w, y1-border_w]], -1, color, 3)

        cv2.putText(image_filtered, predicted_class_name, (x1, y2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        object = class_names[classify_results[0].probs.top1]
        confidence = classify_results[0].probs.top1conf.cpu().numpy()

        if object not in detected_objects or detected_objects[object][confidence] < confidence:
            # For the simulation in vision_pick_and_place.wbt:
            #   x -> [129; 1791] =  [-0.4; 0.4]
            #   y -> [159; 1404] = -[-0.3; 0.3]
            detected_objects[object] = {
                "confidence": confidence,
                "x": 0.8 * (x+x1-border_w - (1791 + 129) / 2) / (1791 - 129),
                "y": - 0.6 * (y+y1-border_w - (1404 + 159) / 2) / (1404 - 159),
                "z": 0.002,
                "theta": math.radians(theta)
            }

    return cv2.cvtColor(image_filtered, cv2.COLOR_RGB2BGR)


while robot.step(timestep) != -1:
    img_array = np.frombuffer(camera.getImage(), dtype=np.uint8)
    img = img_array.reshape((camera.getHeight(), camera.getWidth(), 4))

    if receiver.getQueueLength() > 0:
        message = receiver.getString()
        receiver.nextPacket()

        if message.startswith("image"):
            image = process_image(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer)
            emitter.send("image" + img_base64.decode("utf-8"))
        elif message.startswith("find"):
            object = message[5:]
            if object in detected_objects:
                x = detected_objects[object]["x"]
                y = detected_objects[object]["y"]
                z = detected_objects[object]["z"]
                theta = detected_objects[object]["theta"]

                emitter.send(f"coords_{x:.4f}_{y:.4f}_{z:.4f}_{theta:.4f}")

                del detected_objects[object]
            else:
                emitter.send("not_found")
