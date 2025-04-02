import sys
from controller import Supervisor
import random
import os


OUTPUT_DIR = "images"
CAMERA_NAME = "camera"
OBJECT_1_NAME = "Plate1"
OBJECT_3_NAME = "Plate3"
TABLE_BOUNDS_X = (0.1, 0.15)  # X и Y граници на масата
TABLE_BOUNDS_Y = (-0.15, 0.15)  # X и Y граници на масата

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

supervisor = Supervisor()
camera = supervisor.getDevice(CAMERA_NAME)
camera.enable(100)
obj_1_node = supervisor.getFromDef(OBJECT_1_NAME)
translation_field_1 = obj_1_node.getField("translation")
rotation_field_1 = obj_1_node.getField("rotation")
obj_3_node = supervisor.getFromDef(OBJECT_3_NAME)
translation_field_3 = obj_3_node.getField("translation")
rotation_field_3 = obj_3_node.getField("rotation")


def random_pose(dir=1):
    x = dir * random.uniform(*TABLE_BOUNDS_X)
    y = random.uniform(*TABLE_BOUNDS_Y)
    z = 0.885
    yaw = random.uniform(0, 6.28)
    return (x, y, z, yaw)


iteration = 0
while supervisor.step(100) != -1:
    x, y, z, yaw = random_pose(1 if iteration % 2 == 0 else -1)
    translation_field_1.setSFVec3f([x, y, z])
    rotation_field_1.setSFRotation([0, 0, 1, yaw])

    x, y, z, yaw = random_pose(-1 if iteration % 2 == 0 else 1)
    translation_field_3.setSFVec3f([x, y, z])
    rotation_field_3.setSFRotation([0, 0, 1, yaw])

    supervisor.step(300)

    img_path = os.path.join(
        OUTPUT_DIR, f"image_{iteration:03d}_multiple.jpg")
    camera.getImageArray()
    camera.saveImage(img_path, 100)

    print(f"Image recorded {img_path}")
    iteration += 1
