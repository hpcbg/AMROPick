from controller import Robot

import cv2
import numpy as np
import base64

import amropick

apv = amropick.Vision(detect_model_path="../vision_yolo_controller/detect.pt",
                      classify_model_path="../vision_yolo_controller/classify.pt",
                      class_names=["Plate1", "Plate3"],
                      z_offsets=[0.002, 0.002],
                      class_colors=[(255, 0, 0), (0, 0, 255)])

robot = Robot()

camera = robot.getDevice("camera")
camera.enable(50)

receiver = robot.getDevice("receiver")
receiver.enable(50)

emitter = robot.getDevice("emitter")

timestep = int(robot.getBasicTimeStep())


while robot.step(timestep) != -1:
    img_array = np.frombuffer(camera.getImage(), dtype=np.uint8)
    img = img_array.reshape((camera.getHeight(), camera.getWidth(), 4))

    if receiver.getQueueLength() > 0:
        message = receiver.getString()
        receiver.nextPacket()

        if message.startswith("image"):
            image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            apv.load_image_from_cv2(image)
            apv.process_image()
            _, buffer = cv2.imencode('.jpg', apv.get_processed_image())
            img_base64 = base64.b64encode(buffer)
            emitter.send("image" + img_base64.decode("utf-8"))
        elif message.startswith("find"):
            object = apv.get_object(message[5:])
            if object:
                x = object["x"]
                y = object["y"]
                z = object["z"]
                theta = object["theta"]

                emitter.send(f"coords_{x:.4f}_{y:.4f}_{z:.4f}_{theta:.4f}")
            else:
                emitter.send("not_found")
