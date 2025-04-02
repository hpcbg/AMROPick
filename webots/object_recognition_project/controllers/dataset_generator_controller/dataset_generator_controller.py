from controller import Supervisor
import os
import math
import sys
import random


CAMERA_NAME = "camera"
OBJECT_NAME = sys.argv[1]
TRAIN_IMAGES_COUNT = 55
VAL_IMAGES_COUNT = 8

if not os.path.exists("output/train/images"):
    os.makedirs("output/train/images")

if not os.path.exists("output/val/images"):
    os.makedirs("output/val/images")

supervisor = Supervisor()
camera = supervisor.getDevice(CAMERA_NAME)
camera.enable(100)

obj_node = supervisor.getFromDef(OBJECT_NAME)
translation_field = obj_node.getField("translation")
rotation_field = obj_node.getField("rotation")

iteration = 0
while supervisor.step(100) != -1 and iteration < TRAIN_IMAGES_COUNT + VAL_IMAGES_COUNT:
    translation_field.setSFVec3f([0, 0, 0.884])
    if iteration < TRAIN_IMAGES_COUNT:
        rotation_field.setSFRotation(
            [0, 0, 1, iteration * 2 * math.pi / TRAIN_IMAGES_COUNT])
    else:
        rotation_field.setSFRotation([0, 0, 1, random.uniform(0, 6.28)])

    supervisor.step(300)

    camera.getImageArray()
    if iteration < TRAIN_IMAGES_COUNT:
        camera.saveImage(
            f"output/train/images/image_{sys.argv[1]}_{iteration:03d}.jpg", 100)
    else:
        camera.saveImage(
            f"output/val/images/image_{sys.argv[1]}_{iteration:03d}.jpg", 100)

    iteration += 1

iteration = 0
