from controller import Robot

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFrame, QGroupBox, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import Qt
import sys
import base64
import threading
import cv2
import numpy as np
import time


class ClientRobot(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Robot Vision System Control Panel")

        try:
            screen = QGuiApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()
            self.move(screen_geometry.width() - 700, 80)
        except:
            pass

        main_layout = QVBoxLayout()

        # Environment Frame
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        self.reposition_button = QPushButton("Reposition objects")
        self.reposition_button.clicked.connect(self.send_reposition_request)
        env_layout.addWidget(self.reposition_button)
        env_group.setLayout(env_layout)
        main_layout.addWidget(env_group)

        # Detection Frame
        detect_group = QGroupBox("Object detection")
        detect_layout = QVBoxLayout()

        self.get_image_button = QPushButton("Get image")
        self.get_image_button.clicked.connect(self.send_image_request)
        detect_layout.addWidget(self.get_image_button)

        self.processed_image_label = QLabel("Processed image:")
        detect_layout.addWidget(self.processed_image_label)

        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.StyledPanel)
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(600, 450)
        image_layout.addWidget(self.image_label)
        image_frame.setLayout(image_layout)
        detect_layout.addWidget(image_frame)

        self.find_plate1_button = QPushButton("Find Plate 1")
        self.find_plate1_button.clicked.connect(
            lambda: self.send_request("find_Plate1", self.find_plate1_button))
        self.find_plate1_button.setEnabled(False)
        detect_layout.addWidget(self.find_plate1_button)

        self.find_plate2_button = QPushButton("Find Plate 2")
        self.find_plate2_button.clicked.connect(
            lambda: self.send_request("find_Plate2", self.find_plate2_button))
        self.find_plate2_button.setEnabled(False)
        detect_layout.addWidget(self.find_plate2_button)

        self.find_plate3_button = QPushButton("Find Plate 3")
        self.find_plate3_button.clicked.connect(
            lambda: self.send_request("find_Plate3", self.find_plate3_button))
        self.find_plate3_button.setEnabled(False)
        detect_layout.addWidget(self.find_plate3_button)

        self.find_plate4_button = QPushButton("Find Plate 4")
        self.find_plate4_button.clicked.connect(
            lambda: self.send_request("find_Plate4", self.find_plate4_button))
        self.find_plate4_button.setEnabled(False)
        detect_layout.addWidget(self.find_plate4_button)

        self.find_plate5_button = QPushButton("Find Plate 5")
        self.find_plate5_button.clicked.connect(
            lambda: self.send_request("find_Plate5", self.find_plate5_button))
        self.find_plate5_button.setEnabled(False)
        detect_layout.addWidget(self.find_plate5_button)

        detect_group.setLayout(detect_layout)
        main_layout.addWidget(detect_group)

        # Robot Frame
        robot_group = QGroupBox("Robot control")
        robot_layout = QHBoxLayout()
        self.home_button = QPushButton("Home position")
        self.home_button.clicked.connect(self.send_home_command)
        self.pick_button = QPushButton("Pick object")
        self.pick_button.clicked.connect(self.send_pick_command)
        self.pick_button.setEnabled(False)
        robot_layout.addWidget(self.home_button)
        robot_layout.addWidget(self.pick_button)
        robot_group.setLayout(robot_layout)
        main_layout.addWidget(robot_group)

        # Status Frame
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        self.setLayout(main_layout)

        # Webots communication
        self.robot = Robot()
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(50)

        # Start UI Thread
        self.running = True
        self.listener_thread = threading.Thread(target=self.listen_for_response)
        self.listener_thread.start()

        self.coords = None

        time.sleep(3)
        self.emitter.send("image")

    def disable_buttons(self):
        self.find_plate1_button.setEnabled(False)
        self.find_plate2_button.setEnabled(False)
        self.find_plate3_button.setEnabled(False)
        self.find_plate4_button.setEnabled(False)
        self.find_plate5_button.setEnabled(False)
        self.pick_button.setEnabled(False)

    def enable_find_buttons(self):
        self.find_plate1_button.setEnabled(True)
        self.find_plate2_button.setEnabled(True)
        self.find_plate3_button.setEnabled(True)
        self.find_plate4_button.setEnabled(True)
        self.find_plate5_button.setEnabled(True)

    def send_reposition_request(self):
        self.image_label.setPixmap(QPixmap())
        self.find_plate1_button.setEnabled(False)
        self.find_plate3_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.emitter.send("reposition")

    def send_image_request(self):
        self.status_label.setText(
            "Image request is sent. Waiting for a response...")
        self.emitter.send("image")
        self.disable_buttons()

    def send_request(self, message, button):
        self.status_label.setText(
            f"Object request is sent. Waiting for a response...")
        self.emitter.send(message)
        button.setEnabled(False)
        self.pick_button.setEnabled(False)

    def send_home_command(self):
        self.emitter.send("home")

    def send_pick_command(self):
        self.pick_button.setEnabled(False)
        if self.coords:
            x, y, z, theta = self.coords
            self.emitter.send(f"pick_{x:.4f}_{y:.4f}_{z:.4f}_{theta:.4f}")

    def listen_for_response(self):
        while self.running:
            if self.robot.step(50) == -1:
                break
            if self.receiver.getQueueLength() > 0:
                message = self.receiver.getString()
                self.receiver.nextPacket()

                if message.startswith("image"):
                    try:

                        img_base64 = message[5:]
                        img_bytes = base64.b64decode(img_base64)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if img is None:
                            self.status_label.setText(
                                "Invalid image is received!")
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            height, width, _ = img.shape
                            q_img = QImage(img.data, width, height,
                                           3 * width, QImage.Format_RGB888)
                            pixmap = QPixmap.fromImage(q_img)

                            self.image_label.setPixmap(pixmap.scaled(
                                self.image_label.width(),
                                self.image_label.height(),
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation
                            ))

                            self.status_label.setText(f"Image is received.")
                            self.enable_find_buttons()
                            self.pick_button.setEnabled(False)

                    except Exception as e:
                        self.status_label.setText(
                            f"Image processing error: {e}")
                        self.disable_buttons()
                elif message.startswith("coords"):
                    self.coords = None
                    try:
                        self.coords = tuple(map(float, message.split("_")[1:]))
                        x, y, z, theta = self.coords
                        self.status_label.setText(
                            f"Object found at position ({x:.4f}, {y:.4f}, {z:.4f}) rotated at {theta:.4f} radians.")
                        self.pick_button.setEnabled(True)
                    except Exception as e:
                        self.status_label.setText(
                            f"Error processing coordinates: {e}")
                        self.pick_button.setEnabled(False)
                elif message.startswith("not_found"):
                    self.status_label.setText(
                        "Object is not found. Refresh the image.")
                    self.pick_button.setEnabled(False)

    def closeEvent(self, event):
        self.running = False
        self.listener_thread.join()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClientRobot()
    window.show()
    sys.exit(app.exec_())
