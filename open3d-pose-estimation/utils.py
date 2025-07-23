import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import yaml
import os
from ultralytics import YOLO

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_camera_pose(config):
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(config["camera_pose"]["rotation_deg"]))
    t = np.array(config["camera_pose"]["translation"]).reshape((3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T

def get_robot_pose(config):
    T = np.eye(4)
    T[:3, 3] = np.array(config["robot_pose"]["translation"])
    return T

def extract_masked_pointcloud(mask, depth_frame, color_image, color_intr):
    points = []
    colors = []
    for v in range(mask.shape[0]):
        for u in range(mask.shape[1]):
            if mask[v, u] > 0:
                depth = depth_frame[v, u] / 1000.0
                if depth == 0:
                    continue
                xyz = rs.rs2_deproject_pixel_to_point(color_intr, [u, v], depth)
                points.append(xyz)
                colors.append(color_image[v, u] / 255.0)
    if not points:
        raise RuntimeError("No valid 3D points extracted.")
    return np.array(points), np.array(colors)

def capture_filtered(pipeline, align, n_frames=5, apply_average_filter=False):
    print("[INFO] Warming up sensor...")
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    colorizer = rs.colorizer()
    depth_to_disparity = rs.disparity_transform()
    disparity_to_depth = rs.disparity_transform()

    if apply_average_filter:
        depth_accum = None
        color_accum = None

    for _ in range(n_frames):
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        # depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        if apply_average_filter:
            depth_np = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            if depth_accum is None:
                depth_accum = depth_np
                color_accum = color_image.astype(np.float32)
            else:
                depth_accum += depth_np
                color_accum += color_image.astype(np.float32)

    if apply_average_filter:
        averaged_depth = (depth_accum / n_frames).astype(np.uint16)
        averaged_color = (color_accum / n_frames).astype(np.uint8)
        try:
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        except Exception:
            colorized_depth = np.zeros_like(averaged_color)
        return averaged_depth, averaged_color, colorized_depth

    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    return np.asanyarray(depth_frame.get_data()), np.asanyarray(color_frame.get_data()), colorized_depth

def run_segmentation(model_path, image, valid_classes, confidence=0.5):
    model = YOLO(model_path)
    results = model(image, conf=confidence, overlap_mask=False, retina_masks=True)[0]

    masks = results.masks.data.cpu().numpy() if results.masks else []
    filtered_masks = []
    for i in range(len(masks)):
        mask = (masks[i] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            filtered = np.zeros_like(mask)
            cv2.drawContours(filtered, [largest], -1, 255, cv2.FILLED)
            filtered_masks.append(filtered)
        else:
            filtered_masks.append(mask)

    classes = results.boxes.cls.cpu().numpy() if results.boxes else []
    detections = [(i, results.names[int(cls)]) for i, cls in enumerate(classes)
                  if results.names[int(cls)] in valid_classes]

    return np.array(filtered_masks), detections, results.names

def visualize_detections(image, masks, detections, names, output_path):
    vis_image = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128)
    ]

    for i, (idx, label) in enumerate(detections):
        mask = masks[idx]
        mask = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))
        color = colors[i % len(colors)]
        blended = vis_image.copy()
        blended[mask > 0.5] = vis_image[mask > 0.5] * 0.5 + np.array(color) * 0.5
        vis_image = blended
        cv2.putText(vis_image, f"{i}: {label}", (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imwrite(output_path, vis_image)
    return vis_image

def create_full_pointcloud_from_rgbd(depth_frame, color_frame, color_intr):
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_frame)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False, depth_trunc=3.0)

    pinhole = o3d.camera.PinholeCameraIntrinsic(
        color_intr.width, color_intr.height,
        color_intr.fx, color_intr.fy,
        color_intr.ppx, color_intr.ppy)

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)


def draw_frames(camera_pose=np.eye(4), robot_pose=np.eye(4)):
    def labeled_coordinate_frame(name, transform, size=0.1, color=[1, 0, 0]):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(transform)
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.1)
        marker.paint_uniform_color(color)
        marker.translate(transform[:3, 3] + np.array([0, 0, size * 1.5]))
        return [frame, marker]

    camera_frame = labeled_coordinate_frame("Camera", camera_pose, color=[0, 1, 0])
    robot_frame = labeled_coordinate_frame("Robot", robot_pose, color=[0, 0, 1])
    return camera_frame + robot_frame