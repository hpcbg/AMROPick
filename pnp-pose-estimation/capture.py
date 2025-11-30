import cv2
import time
from oakd_camera import OAKDCamera


# Params
DEVICE_ID = 0
PATH = './capture'
OAK_D = True

if OAK_D:
    cap = OAKDCamera(rgb_resolution="4K", fps=30, preview_size=(3840, 2160))
    cap.open()
else:
    cap = cv2.VideoCapture(DEVICE_ID)

if not cap.isOpened():
    print('Cannot open camera!')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Cannot retrieve camera frame!')
        break

    frame2 = frame.copy()

    if OAK_D:
        frame2 = cv2.resize(frame2, (960, 540))

    cv2.putText(frame2, 'C=Capture, Esc=Quit', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Camera', frame2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = f'{PATH}/capture_{int(time.time())}.png'
        cv2.imwrite(filename, frame)
        print(f'Image saved as {filename}')

    elif key == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
