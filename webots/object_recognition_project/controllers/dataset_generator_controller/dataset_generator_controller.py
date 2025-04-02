from controller import Supervisor
import os
import math
import sys


OUTPUT_DIR = "images"
CAMERA_NAME = "camera"
OBJECT_NAME = sys.argv[1]
IMAGES_COUNT = 100

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

supervisor = Supervisor()
camera = supervisor.getDevice(CAMERA_NAME)
camera.enable(100)

obj_node = supervisor.getFromDef(OBJECT_NAME)
translation_field = obj_node.getField("translation")
rotation_field = obj_node.getField("rotation")

iteration = 0
while supervisor.step(100) != -1 and iteration < IMAGES_COUNT:
    translation_field.setSFVec3f([0, 0, 0.884])
    rotation_field.setSFRotation([0, 0, 1, iteration * 2 * math.pi / 100])

    supervisor.step(300)

    img_path = os.path.join(
        OUTPUT_DIR, f"image_{sys.argv[1]}_{iteration:03d}.jpg")
    camera.getImageArray()
    camera.saveImage(img_path, 100)

    print(f"Image recorded {img_path}")
    iteration += 1
