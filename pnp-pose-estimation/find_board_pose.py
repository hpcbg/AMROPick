import cv2
import numpy as np

# Params
CALIB_FILE = './calibration/calib_data.npz'  # camera calibration
CHESSBOARD = (8, 6)  # number of internal corners (cols, rows)
SQUARE_SIZE = 25.0  # mm
DEVICE_ID = 0

# Load calibration
data = np.load(CALIB_FILE)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']
print(f'Camera calibration is loaded from {CALIB_FILE}')

# 3D points of the board (Z=0)
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 3D vectors of the axis (length of 3 squares)
axis = np.float32([
    [3*SQUARE_SIZE, 0, 0],        # X – red
    [0, 3*SQUARE_SIZE, 0],        # Y – green
    [0, 0, -3*SQUARE_SIZE]        # Z – blue
])

# 3D points for the cube (side with length of 3 squares)
cube_size = 3 * SQUARE_SIZE
cube_points = np.float32([
    [0, 0, 0],
    [0, cube_size, 0],
    [cube_size, cube_size, 0],
    [cube_size, 0, 0],
    [0, 0, -cube_size],
    [0, cube_size, -cube_size],
    [cube_size, cube_size, -cube_size],
    [cube_size, 0, -cube_size]
])


def to_pt(arr):
    """Helper: convert (x,y) float to (int(x), int(y)) for cv2.line."""
    a = np.ravel(arr).astype(float)
    return (int(round(a[0])), int(round(a[1])))


# Draw coordinate frame
def draw_axes(img, corners, imgpts):
    corner = to_pt(corners[0])
    img = cv2.line(img, corner, to_pt(imgpts[0]), (0, 0, 255), 4)   # X red
    img = cv2.line(img, corner, to_pt(imgpts[1]), (0, 255, 0), 4)   # Y green
    img = cv2.line(img, corner, to_pt(imgpts[2]), (255, 0, 0), 4)   # Z blue
    return img


# Draw a cube
def draw_cube(img, imgpts):
    imgpts_i = np.int32(imgpts).reshape(-1, 2)
    # Bottom side
    img = cv2.drawContours(img, [imgpts_i[:4]], -1, (0, 255, 0), 2)
    # Vertical lines
    for i in range(4):
        img = cv2.line(img, tuple(imgpts_i[i]), tuple(
            imgpts_i[i+4]), (255, 0, 0), 2)
    # Top side
    img = cv2.drawContours(img, [imgpts_i[4:]], -1, (0, 0, 255), 2)
    return img


cap = cv2.VideoCapture(DEVICE_ID)
if not cap.isOpened():
    print(f'Cannot open camera with ID {DEVICE_ID}')
    exit()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)

        # Get the pose with solvePnP
        ok, rvec, tvec = cv2.solvePnP(
            objp, corners_refined, camera_matrix, dist_coeffs)
        if ok:
            # Axes
            imgpts_axis, _ = cv2.projectPoints(
                axis, rvec, tvec, camera_matrix, dist_coeffs)
            frame = draw_axes(frame, corners_refined, imgpts_axis)

            # Cube
            imgpts_cube, _ = cv2.projectPoints(
                cube_points, rvec, tvec, camera_matrix, dist_coeffs)
            frame = draw_cube(frame, imgpts_cube)

            # Translation vector
            tx, ty, tz = tvec.flatten()
            text = f'Tvec (mm): X={tx:.1f}, Y={ty:.1f}, Z={tz:.1f}'
            cv2.putText(frame, text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # Corners
        cv2.drawChessboardCorners(frame, CHESSBOARD, corners_refined, found)

    cv2.putText(frame, 'Esc=Quit', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Pose Estimation (Axes + Cube)', frame)

    if (cv2.waitKey(1) & 0xFF) == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
