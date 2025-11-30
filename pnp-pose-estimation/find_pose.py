import cv2
import numpy as np
import traceback
from math import atan2, degrees
import open3d as o3d
import threading
from oakd_camera import OAKDCamera

# Paras
OAK_D = True
# CALIB_FILE = './calibration/calib_data.npz'  # camera calibration
CALIB_FILE = './calibration/oakd/calib_data_oakd.npz'  # camera calibration
DEVICE_ID = 0  # camera device ID
if OAK_D:
    TOP, BOTTOM, LEFT, RIGHT = 200, 1800, 200, 3400
else:
    TOP, BOTTOM, LEFT, RIGHT = 100, 420, 30, 520
THRESHOLD = 90
MAX_IOU_ERROR = 0.1
obj_pts = np.array([  # Plate 3 3D Printed
    [0,  0, 0],
    [0, 50, 0],
    [150, 50, 0],
    [150, 25, 0],
], dtype=np.float32)
# obj_pts = np.array([  # Plate 3
#     [0, 0, 0],
#     [0, 35, 0],
#     [100, 35, 0],
#     [100, 19, 0],
# ], dtype=np.float32)
# obj_pts = np.array([ # Plate 1
#     [0,0, 0],
#     [0,18, 0],
#     [100,18, 0],
#     [100,0, 0]
# ], dtype=np.float32)


data = np.load(CALIB_FILE)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']
prev_T = np.eye(4)
prev_rvec = None
prev_tvec = None


class CameraThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if OAK_D:
            self.cap = OAKDCamera(rgb_resolution="4K", fps=5,
                                  preview_size=(3840, 2160))
            self.cap.open()
        else:
            self = cv2.VideoCapture(DEVICE_ID)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# Open3D
board_w = 100
board_h = 18
thickness = 2
vis = o3d.visualization.Visualizer()
vis.create_window("Pose", width=960, height=720)
cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
vis.add_geometry(cam_frame)
board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
board = o3d.geometry.TriangleMesh.create_box(
    width=board_w, height=board_h, depth=thickness)
board.translate([-board_w/2, -board_h/2, -thickness/2])  # center in (0,0,0)
board.compute_vertex_normals()
board.paint_uniform_color([0, 0, 0])
center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
center_sphere.paint_uniform_color([0, 1, 0])
vis.add_geometry(center_sphere)
vis.add_geometry(board_frame)
vis.add_geometry(board)


# Retrieve RPY (ZYX) from R
def rotationMatrixToRPY(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(-R[2, 0], sy)
        z = atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0
    return degrees(x), degrees(y), degrees(z)


# Generate all sequences of points
def rotations_and_flips(points):
    n = len(points)
    for shift in range(n):
        yield np.roll(points, shift, axis=0)
        yield np.flip(np.roll(points, shift, axis=0), axis=0)


cap = CameraThread(DEVICE_ID)


while True:
    frame = cap.read()
    if frame is None:
        continue

    processed = frame.copy()
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # undistorted = cv2.undistort(frame_grayscale, camera_matrix, dist_coeffs)

    cropped = 255 * np.ones(frame_grayscale.shape, dtype=np.uint8)
    cropped[TOP:BOTTOM, LEFT:RIGHT] = frame_grayscale[TOP:BOTTOM, LEFT:RIGHT]

    _, binary = cv2.threshold(cropped, THRESHOLD, 255, cv2.THRESH_BINARY)
    binary = 255 - binary

    if OAK_D:
        cv2.imshow('binary', cv2.resize(binary, (960, 540)))
    else:
        cv2.imshow('binary', binary)
    try:
        cnts, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            img_pts = approx.reshape(-1, 2).astype(np.float32)
            contour_to_draw = approx  # shape: (N,1,2)
            contour = np.zeros(frame_grayscale.shape, dtype=np.uint8)
            cv2.drawContours(contour, [contour_to_draw], -1, 255, -1)
            if OAK_D:
                cv2.imshow('contour', cv2.resize(contour, (960, 540)))
            else:
                cv2.imshow('contour', contour)

            best_error = float('inf')
            best_rvec, best_tvec = None, None
            for candidate_points in rotations_and_flips(img_pts):
                success = False
                try:
                    if prev_rvec is None or prev_tvec is None:
                        success, rvec, tvec, inliers = cv2.solvePnPRansac(
                            objectPoints=obj_pts,
                            imagePoints=candidate_points,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            reprojectionError=0.5,
                            confidence=0.99,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                    else:
                        success, rvec, tvec, inliers = cv2.solvePnPRansac(
                            objectPoints=obj_pts,
                            imagePoints=candidate_points,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            rvec=prev_rvec,
                            tvec=prev_tvec,
                            useExtrinsicGuess=True,
                            reprojectionError=0.5,
                            confidence=0.99,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                except Exception as x:
                    pass
                    # traceback.print_exception(x)
                if success:
                    proj, _ = cv2.projectPoints(
                        obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
                    proj = proj.squeeze().astype(np.int32)
                    mask_proj = np.zeros(binary.shape, dtype=np.uint8)
                    cv2.fillPoly(mask_proj, [proj], 255)
                    intersection = np.logical_and(
                        mask_proj > 0, contour > 0).sum()
                    union = np.logical_or(mask_proj > 0, contour > 0).sum()
                    iou = intersection / union if union > 0 else 0
                    error = 1 - iou
                    if error < best_error:
                        best_error = error
                        best_rvec, best_tvec = rvec.copy(), tvec.copy()

            rvec = best_rvec
            tvec = best_tvec
            prev_rvec = rvec
            prev_tvec = tvec

            if best_error < MAX_IOU_ERROR:
                # Text info
                R, _ = cv2.Rodrigues(rvec)
                point_in_cam = R @ np.mean(obj_pts,
                                           axis=0).reshape(3, 1) + tvec
                tx, ty, tz = tuple(point_in_cam.flatten())
                text = f'Tvec (mm): X={tx:.1f}, Y={ty:.1f}, Z={tz:.1f}'
                cv2.putText(processed, text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            3 if OAK_D else 0.6, (0, 0, 0), 2)
                roll, pitch, yaw = rotationMatrixToRPY(R)
                text = f'Tvec (mm): X={tx:.1f}, Y={ty:.1f}, Z={tz:.1f}'
                cv2.putText(processed, f'Roll: {roll:4.0f}, Pitch: {pitch:4.0f}, Yaw: {yaw:4.0f}',
                            (15, 150 if OAK_D else 90), cv2.FONT_HERSHEY_SIMPLEX, 3 if OAK_D else 0.6, (0, 0, 0), 2)

                # Object overlay
                img_points_proj, _ = cv2.projectPoints(
                    obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
                pts = np.int32(img_points_proj).reshape(-1, 2)
                overlay = processed.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                alpha = 0.4
                processed = cv2.addWeighted(
                    overlay, alpha, processed, 1 - alpha, 0)

                # Open3D
                R, _ = cv2.Rodrigues(rvec)
                points_3d_cam = (R @ obj_pts.T + tvec).T
                obj_center = np.mean(obj_pts, axis=0)
                tvec_center = R @ obj_center.reshape(3, 1) + tvec
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec_center.flatten()
                inv_prev = np.linalg.inv(prev_T)
                board_frame.transform(inv_prev)
                board.transform(inv_prev)
                center_sphere.transform(inv_prev)
                board_frame.transform(T)
                board.transform(T)
                center_sphere.transform(T)
                prev_T = T
                vis.update_geometry(center_sphere)
                vis.update_geometry(board_frame)
                vis.update_geometry(board)
    except Exception as x:
        pass
        # traceback.print_exception(x)

    if OAK_D:
        cv2.imshow('processed', cv2.resize(processed, (960, 540)))
    else:
        cv2.imshow('processed', processed)

    if (cv2.waitKey(1) & 0xFF) == 27:  # Esc
        break

    # Update Open3D
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(1000.0)
    ctr.set_constant_z_near(0.01)
    vis.poll_events()
    vis.update_renderer()

cap.stop()
cv2.destroyAllWindows()
vis.destroy_window()
