import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import yaml
from ultralytics import YOLO
from utils import extract_masked_pointcloud, capture_filtered
from realsense_utils import setup_pipeline
from run_icp_alignment import run_alignment
import os

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

def main():
    config = load_config()

    pipeline, align, profile = setup_pipeline()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_profile.get_intrinsics()

    camera_pose = get_camera_pose(config)
    robot_pose = np.eye(4)

    print("[INFO] Capturing frame...")
    os.makedirs("output_images", exist_ok=True)
    depth_frame, color_image, depth_vis = capture_filtered(pipeline, align)
    cv2.imwrite(os.path.join("output_images", "captured_rgb.png"), color_image)

    print("[INFO] Running segmentation...")
    model = YOLO(config["paths"]["model_weights_path"])
    results = model(color_image, conf=config["segmentation"]["confidence_threshold"], overlap_mask=False)[0]

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
    filtered_masks = np.array(filtered_masks)

    classes = results.boxes.cls.cpu().numpy() if results.boxes else []
    detections = [(i, results.names[int(cls)]) for i, cls in enumerate(classes)
                  if results.names[int(cls)] in config["valid_classes"]]

    if not detections:
        print("[ERROR] No valid objects detected.")
        return

    vis_image = color_image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128)
    ]

    for i, (idx, label) in enumerate(detections):
        mask = filtered_masks[idx]
        mask = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))
        color = colors[i % len(colors)]
        blended = vis_image.copy()
        blended[mask > 0.5] = vis_image[mask > 0.5] * 0.5 + np.array(color) * 0.5
        vis_image = blended
        cv2.putText(vis_image, f"{i}: {label}", (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imwrite(os.path.join("output_images", "detection_preview.png"), vis_image)
    cv2.imshow("Detected Objects", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Detected valid objects:")
    for i, (idx, label) in enumerate(detections):
        print(f"{i}: {label}")

    choice = int(input("Choose object index: "))
    obj_idx = detections[choice][0]
    class_label = detections[choice][1]
    model_path = config["model_mapping"].get(class_label)
    if model_path is None:
        print(f"[ERROR] No model defined for class {class_label}")
        return

    mask = filtered_masks[obj_idx]
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join("output_images", f"filtered_mask_{obj_idx}.png"), mask)

    points, colors = extract_masked_pointcloud(mask, depth_frame, color_image, color_intr)
    cut_pcd = o3d.geometry.PointCloud()
    cut_pcd.points = o3d.utility.Vector3dVector(points)
    cut_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(config["paths"]["cut_scene_output"], cut_pcd)

    result = run_alignment(
        model_path=model_path,
        scene_path=config["paths"]["cut_scene_output"],
        voxel_size=config["alignment"]["voxel_size"],
        init_translation=config["alignment"]["init_translation"],
        init_rotation=config["alignment"]["init_rotation_deg"],
        visualize=config["alignment"]["visualize"],
        skip_ransac=config["alignment"]["skip_ransac"],
        icp_threshold=config["alignment"]["icp_threshold"]
    )

    T_model_to_object = result.transformation
    T_model_to_robot = T_model_to_object

    print("[INFO] Transformation from model to robot base:")
    print(T_model_to_robot)

    model_pcd = o3d.io.read_point_cloud(model_path)
    model_pcd.transform(T_model_to_robot)

    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_frame)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False, depth_trunc=3.0)
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        color_intr.width, color_intr.height,
        color_intr.fx, color_intr.fy,
        color_intr.ppx, color_intr.ppy)
    full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)

    o3d.visualization.draw_geometries(
        [full_pcd, model_pcd] + draw_frames(camera_pose, robot_pose)
    )

    pipeline.stop()

if __name__ == "__main__":
    main()