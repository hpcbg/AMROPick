import cv2
import numpy as np

# Params
DEVICE_ID = 0
data = np.load('./calibration/calib_data.npz')
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

cap = cv2.VideoCapture(DEVICE_ID)

if not cap.isOpened():
    print('Cannot open camera')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Show
    cv2.imshow('Original', frame)
    cv2.imshow('Undistorted', undistorted)

    # Quit on Esc
    if cv2.waitKey(1) & 0xFF == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
