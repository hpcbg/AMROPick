import cv2
import numpy as np
from oakd_camera import OAKDCamera

# Params
CHESSBOARD = (8, 6)  # internal corners (cols, rows)
SQUARE_SIZE = 25.0   # square size (mm)
MIN_SAMPLES = 12     # min samples
SAVE_PATH = "calib_data.npz"
DEVICE_ID = 0        # camera ID
OAK_D = True

# 3D points
objp = np.zeros((CHESSBOARD[1] * CHESSBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

if OAK_D:
    cap = OAKDCamera(rgb_resolution="4K", fps=30, preview_size=(3840, 2160))
    cap.open()
else:
    cap = cv2.VideoCapture(DEVICE_ID)

if not cap.isOpened():
    print(f'Cannot open device with ID: {DEVICE_ID}')
    exit()

calibrated = False
undistort_live = False
camera_matrix = None
dist_coeffs = None
mapx = None
mapy = None
roi = None
img_size = None
rms_err = None


def draw_info(img, text_lines):
    y = 30
    for line in text_lines:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        y += 25


print('C=Capture frame, K=Calibrate, U=Toggle undistort, Esc=Quit')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = (gray.shape[1], gray.shape[0])

    found, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, CHESSBOARD, corners_refined, found)

    # Undistort live with crop
    if calibrated and undistort_live:
        undistorted = cv2.remap(
            frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        # return original shape
        display = cv2.resize(undistorted, img_size)
    else:
        display = frame.copy()

    if OAK_D:
        display = cv2.resize(display, (960, 540))

    draw_info(display, [
        f'Frames captured: {len(objpoints)}',
        f'Undistort: {'ON' if undistort_live else 'OFF'}',
        'C=Capture frame, K=Calibrate, U=Toggle undistort, Esc=Quit',
        f'RMS error: {rms_err:.4f}' if calibrated else ''
    ])

    cv2.imshow('Camera Calibration', display)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Esc
        break
    elif key in (ord('c'), ord('C')):
        if found:
            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)
            print(f'Captured frame #{len(objpoints)}')
        else:
            print('Board not found - frame skipped.')
    elif key in (ord('k'), ord('K')):
        if len(objpoints) >= MIN_SAMPLES:
            print('Calibration...')
            rms_err, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )
            print(f'RMS error: {rms_err:.4f}')

            # New matrix, ROI, alpha=0 → no black zones
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, img_size, 0, img_size
            )

            # Undistort map for fast processing
            mapx, mapy = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None, new_camera_matrix, img_size, cv2.CV_32FC1
            )

            # Save calibration
            np.savez(SAVE_PATH,
                     camera_matrix=camera_matrix,
                     dist_coeffs=dist_coeffs,
                     image_size=img_size,
                     rms=rms_err,
                     chessboard_cols=CHESSBOARD[0],
                     chessboard_rows=CHESSBOARD[1],
                     square_size=SQUARE_SIZE,
                     samples=len(objpoints))
            print(f'Camera calibration saved in {SAVE_PATH}')
            calibrated = True
        else:
            print(f'At least {MIN_SAMPLES} frames are needed for calibration.')
    elif key in (ord('u'), ord('U')):
        if calibrated:
            undistort_live = not undistort_live
        else:
            print('First calibrate the camera (Press K).')

cap.release()
cv2.destroyAllWindows()
