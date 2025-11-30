import cv2
import numpy as np
import open3d as o3d


# Params
PATTERN_SIZE = (8, 6)   # number of internal corners (cols, rows)
SQUARE_SIZE = 25.0      # mm
DEVICE_ID = 0           # camera device id

data = np.load('./calibration/calib_data.npz')
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']


# 3D points of the board
objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # in mm

# Board dimensions (real board have 1 more square)
board_w = (PATTERN_SIZE[0]-1) * SQUARE_SIZE
board_h = (PATTERN_SIZE[1]-1) * SQUARE_SIZE

# Open3D Scene
vis = o3d.visualization.Visualizer()
vis.create_window('Open3D Pose', width=960, height=720)

# Camera coordinate frame (static in (0,0,0))
cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
vis.add_geometry(cam_frame)

# Board coordinate frame (movable)
board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)

# The board (rectangle), centered in (0,0,0) in local board coordinates
thickness = 1.0
board = o3d.geometry.TriangleMesh.create_box(
    width=board_w, height=board_h, depth=thickness)
board.translate([-board_w/2, -board_h/2, -thickness/2])  # център в (0,0,0)
board.compute_vertex_normals()
board.paint_uniform_color([0, 0, 0])

vis.add_geometry(board_frame)
vis.add_geometry(board)

# Previous transfrom
prev_T = np.eye(4)

cap = cv2.VideoCapture(DEVICE_ID)

# Criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print('Press Esc in OpenCV window to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, PATTERN_SIZE,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    overlay = frame.copy()

    if found:
        # precise corners
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Pose
        ok, rvec, tvec = cv2.solvePnP(
            objp, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            # Draw axis over 2D
            axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, -50]])
            imgpts, _ = cv2.projectPoints(
                axis, rvec, tvec, camera_matrix, dist_coeffs)
            corner = tuple(corners[0].ravel().astype(int))
            cv2.line(overlay, corner, tuple(
                imgpts[0].ravel().astype(int)), (0, 0, 255), 3)  # X red
            cv2.line(overlay, corner, tuple(
                imgpts[1].ravel().astype(int)), (0, 255, 0), 3)  # Y green
            cv2.line(overlay, corner, tuple(
                imgpts[2].ravel().astype(int)), (255, 0, 0), 3)  # Z blue

            # Distance to board center (camera -> object)
            dist_to_board = float(np.linalg.norm(tvec))
            cv2.putText(overlay, f'Dist: {dist_to_board:.1f} mm', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # ---------- 4x4 Transform for Open3D ----------
            R, _ = cv2.Rodrigues(rvec)  # 3x3
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            # Update pose of board_frame and board:
            # 1) Inverse the previous one
            inv_prev = np.linalg.inv(prev_T)
            board_frame.transform(inv_prev)
            board.transform(inv_prev)
            # 2) apply new transform
            board_frame.transform(T)
            board.transform(T)
            prev_T = T

            vis.update_geometry(board_frame)
            vis.update_geometry(board)

    # Update Open3D render
    vis.poll_events()
    vis.update_renderer()

    # Show 2D overlay
    cv2.imshow('Chessboard Pose (OpenCV)', overlay)
    if (cv2.waitKey(1) & 0xFF) == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
