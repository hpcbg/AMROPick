import cv2
import numpy as np


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
        print("Object not found")
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
