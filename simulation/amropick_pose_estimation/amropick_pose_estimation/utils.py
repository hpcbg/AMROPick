# amropick_pose_estimation/utils.py

import cv2
import numpy as np
import open3d as o3d
import yaml
import os
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

MAX_ANGLES_TO_STORE = 10 # Should match the value used during feature creation

def clean_mask(mask):
    """Keeps only the largest connected component of a binary mask."""
    num_labels, labels_matrix, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels_matrix == largest_label).astype(np.uint8) * 255
    return mask

def get_angles(poly):
    angles = []
    points = poly.reshape(-1, 2)
    num_points = len(points)
    if num_points < 3: return []
    for i in range(num_points):
        p_prev, p_curr, p_next = points[i - 1], points[i], points[(i + 1) % num_points]
        vec_a, vec_b = p_prev - p_curr, p_next - p_curr
        dot_product = np.dot(vec_a, vec_b)
        mag_a, mag_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
        if mag_a * mag_b == 0: continue
        angle_rad = np.arccos(np.clip(dot_product / (mag_a * mag_b), -1.0, 1.0))
        angles.append(np.degrees(angle_rad))
    return angles

def extract_features(mask):
    """Calculates the full, advanced set of geometric features."""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 50: return None

    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    
    min_area_rect = cv2.minAreaRect(contour)
    min_rect_w, min_rect_h = min_area_rect[1]
    rotation_angle = min_area_rect[2]
    min_rect_area = min_rect_w * min_rect_h
    rectangularity = area / min_rect_area if min_rect_area > 0 else 0

    epsilon = 0.02 * perimeter
    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
    angles = get_angles(approx_poly)
    sorted_angles = sorted(angles, reverse=True)
    normalized_angles = np.zeros(MAX_ANGLES_TO_STORE)
    num_angles_to_add = min(len(sorted_angles), MAX_ANGLES_TO_STORE)
    normalized_angles[:num_angles_to_add] = sorted_angles[:num_angles_to_add]
    
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    num_vertices = len(approx_poly)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i])) if hu_moments[i] != 0 else 0
    num_holes = sum(1 for h in hierarchy[0] if h[3] != -1)

    feature_dict = {
        'area': area, 'perimeter': perimeter, 'aspect_ratio': aspect_ratio,
        'extent': extent, 'solidity': solidity, 'circularity': circularity,
        'min_rect_w': min_rect_w, 'min_rect_h': min_rect_h, 
        'rotation_angle': rotation_angle, 'rectangularity': rectangularity,
        'num_vertices': num_vertices, 'num_holes': num_holes
    }
    for i, hu in enumerate(hu_moments):
        feature_dict[f'hu_moment_{i+1}'] = hu
    for i, angle in enumerate(normalized_angles):
        feature_dict[f'angle_{i+1}'] = angle
        
    feature_dict['robust_aspect_ratio'] = max(min_rect_w, min_rect_h) / min(min_rect_w, min_rect_h) if min(min_rect_w, min_rect_h) > 0 else 0
    
        
    return feature_dict


def load_config(node=None, config_file='config/config.yaml'):
    """
    Loads a config file from a ROS 2 package and resolves
    all file paths within it to be absolute.
    """
    package_name = 'amropick_pose_estimation'
    try:
        # Find the package's share directory
        package_share_dir = get_package_share_directory(package_name)
    except Exception as e:
        if node:
            node.get_logger().error(f"Could not find package '{package_name}'. Is it built and sourced? Error: {e}")
        return None

    # Get the absolute path to the config file
    config_path = os.path.join(package_share_dir, config_file)

    if not os.path.exists(config_path):
        if node:
            node.get_logger().error(f"Config file not found at: {config_path}")
        return None

    if node:
        node.get_logger().info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Resolve file paths inside the config ---
    # This makes all paths in the config absolute, relative to the share directory

    # 1. Resolve weights path
    weights_path = cfg['paths']['model_weights_path']
    cfg['paths']['model_weights_path'] = os.path.join(package_share_dir, weights_path)
    # sam_cfg: "weights/sam2.1_hiera_large.pt"
    # sam_ckpt: "config/sam2.1_hiera_l.yaml"
    sam_cfg_path = cfg['paths']['sam_cfg']
    cfg['paths']['sam_cfg'] = os.path.join(package_share_dir, sam_cfg_path)
    sam_ckpt_path = cfg['paths']['sam_ckpt']
    cfg['paths']['sam_ckpt'] = os.path.join(package_share_dir, sam_ckpt_path)
    
    # 2. Resolve all model paths in model_mapping
    for key, model_path in cfg['model_mapping'].items():
        cfg['model_mapping'][key] = os.path.join(package_share_dir, model_path)

    for key, model_path in cfg['paths']['classifier'].items():
        cfg['paths']['classifier'][key] = os.path.join(package_share_dir, model_path)
    
    # 3. Resolve intermediate results path (make it absolute from CWD)
    results_path = cfg['paths']['intermediate_results']
    cfg['paths']['intermediate_results'] = os.path.abspath(results_path)

    if node:
        node.get_logger().info("Config loaded and paths resolved.")
    return cfg


def deproject_pixel_to_point_from_info(camera_info: CameraInfo, pixel: list, depth: float) -> list:
    """
    Re-implementation of pyrealsense2.rs2_deproject_pixel_to_point
    using a sensor_msgs/CameraInfo message.
    """
    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    z = depth
    x = (pixel[0] - cx) * z / fx
    y = (pixel[1] - cy) * z / fy
    
    return [x, y, z]

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

def extract_masked_pointcloud(mask, depth_frame, color_image, camera_info: CameraInfo):
    """
    MODIFIED: Takes CameraInfo message instead of rs.intrinsics
    """
    points = []
    colors = []
    depth_scale = 0.001 # Assuming depth_frame is 16UC1 (millimeters)

    for v in range(mask.shape[0]):
        for u in range(mask.shape[1]):
            if mask[v, u] > 0:
                depth = depth_frame[v, u] * depth_scale
                if depth == 0:
                    continue
                
                # Use our new ROS-based deprojection function
                xyz = deproject_pixel_to_point_from_info(camera_info, [u, v], depth)
                
                points.append(xyz)
                colors.append(color_image[v, u] / 255.0)
    
    if not points:
        raise RuntimeError("No valid 3D points extracted.")
    return np.array(points), np.array(colors)

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
                  ]

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


def create_full_pointcloud_from_rgbd(depth_frame, color_frame, camera_info: CameraInfo):
    """
    MODIFIED: Takes CameraInfo message instead of rs.intrinsics
    """
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_frame)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False, depth_trunc=3.0)

    pinhole = o3d.camera.PinholeCameraIntrinsic(
        camera_info.width, camera_info.height,
        camera_info.k[0],  # fx
        camera_info.k[4],  # fy
        camera_info.k[2],  # cx
        camera_info.k[5]   # cy
    )
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


def select_grasp_point_from_model(model_path, visualize_all=False, resampling=5000, diameter_rescale=100):
    import open3d as o3d
    import numpy as np

    mesh = o3d.io.read_triangle_mesh(model_path)

    if len(mesh.triangles) == 0:
        print("[WARN] Model has no triangles. Trying to read as point cloud.")
        pcd = o3d.io.read_point_cloud(model_path)
    else:
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(resampling)
        # mesh.compute_vertex_normals()
        # mesh.orient_triangles()
        # mesh.orient_normals_consistent_tangent_plane(100)
        # pcd = mesh.sample_points_poisson_disk(resampling)
        # pcd.estimate_normals()
        # pcd.orient_normals_consistent_tangent_plane(100)

    if visualize_all:
        print(f"vis: {visualize_all}")
        o3d.visualization.draw_geometries([pcd], window_name="Resampled point cloud")

    diameter = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    camera = [0, 0, diameter]
    diameter_scaled = diameter * diameter_rescale
    _, pt_map = pcd.hidden_point_removal(camera, diameter_scaled)
    pcd_onesided = pcd.select_by_index(pt_map)

    if visualize_all:
        o3d.visualization.draw_geometries([pcd_onesided], window_name="Visible surface")

    print("Pick grasp point(s) with Shift + Left Click, then close the window.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd_onesided)
    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    points = np.asarray(pcd_onesided.points)
    normals = np.asarray(pcd_onesided.normals)
    return points[picked_indices], normals[picked_indices]

def create_grasp_frame(grasp_point, grasp_normal, size=0.07):
    import open3d as o3d
    import numpy as np

    z_axis = -grasp_normal / np.linalg.norm(grasp_normal)
    tmp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(tmp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = grasp_point
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size).transform(T)