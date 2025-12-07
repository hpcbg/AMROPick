import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import yaml
import os
from ultralytics import YOLO

def load_config(path="config.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)

    with open(full_path, "r") as f:
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


def capture_filtered(pipeline, align, n_frames=1, use_disparity=False):
    config = load_config()
    print("[INFO] Warming up sensor...")
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Spatial sharpening values
    spatial.set_option(rs.option.filter_magnitude, config["camera"]["post_processing"]["spatial"]["magnitude"])  # 1–5
    spatial.set_option(rs.option.filter_smooth_alpha, config["camera"]["post_processing"]["spatial"]["smooth_alpha"])  # sharpening factor (0–1)
    spatial.set_option(rs.option.filter_smooth_delta, config["camera"]["post_processing"]["spatial"]["smooth_delta"])  # edge threshold

    temporal.set_option(rs.option.filter_smooth_alpha, config["camera"]["post_processing"]["temporal"]["smooth_alpha"])  # sharpening factor (0–1)
    temporal.set_option(rs.option.filter_smooth_delta, config["camera"]["post_processing"]["temporal"]["smooth_delta"])  # edge threshold

    depth_frame = depth_to_disparity.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = disparity_to_depth.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    colorizer = rs.colorizer()
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

def project_points_to_image(points_3d, intrinsics):
    """
    Projects Nx3 3D points in camera frame onto the RGB image.

    points_3d: (N,3) numpy array in *camera* coordinates
    intrinsics: open3d.camera.PinholeCameraIntrinsic
    """
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    z = np.where(z == 0, 1e-6, z)        # avoid division by zero

    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    return np.stack([u, v], axis=1)

def draw_model_on_image(image, mesh, T_model_cam, intrinsics, color=(0,255,0)):
    """
    Draws the projected mesh edges of a CAD model onto an RGB image.

    image: numpy RGB image (HxWx3)
    mesh:  open3d.geometry.TriangleMesh
    T_model_cam: 4x4 transformation (model → camera)
    intrinsics: camera intrinsics
    """

    img = image.copy()

    # Convert mesh to lineset so we can draw edges
    mesh.compute_vertex_normals()
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

    # Get vertices (Nx3)
    verts = np.asarray(lineset.points)

    # Transform vertices from model → camera frame
    verts_h = np.hstack((verts, np.ones((verts.shape[0], 1))))
    verts_cam = (T_model_cam @ verts_h.T).T[:, :3]


    # --- DEBUG Z-values BEFORE projection ---
    print("\n--- Projection Debug ---")
    print("Z min / max:", verts_cam[:, 2].min(), verts_cam[:, 2].max())

    # Now compute projections (must come before pix-debug)
    pix = project_points_to_image(verts_cam, intrinsics)

    # image resolution
    h, w = img.shape[:2]

    # --- DEBUG: PIXEL RANGE ---
    print("Projected u min/max:", pix[:, 0].min(), pix[:, 0].max())
    print("Projected v min/max:", pix[:, 1].min(), pix[:, 1].max())

    # How many points fall inside the RGB frame?
    inside_mask = (
        (pix[:, 0] >= 0) & (pix[:, 0] < w) &
        (pix[:, 1] >= 0) & (pix[:, 1] < h)
    )
    
    print("Points inside image:", inside_mask.sum(), "/", len(pix))

    # Draw 20 debug points
    for (u, v) in pix[:20]:
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img, (u, v), 4, (0, 0, 255), -1)

    print("--- End Projection Debug ---\n")

    # Project to image
    pix = project_points_to_image(verts_cam, intrinsics)

    # Draw lines on image
    for (i0, i1) in lineset.lines:
        u0, v0 = pix[i0]
        u1, v1 = pix[i1]

        if 0 <= u0 < img.shape[1] and 0 <= u1 < img.shape[1] and \
           0 <= v0 < img.shape[0] and 0 <= v1 < img.shape[0]:
            cv2.line(img, (u0, v0), (u1, v1), color, 2)

    return img


def capture_from_file(depth_path, color_path, depth_scale=1.0):
    """
    Loads a depth and RGB image from files and returns them
    in the same format as capture_frames().

    depth_path: path to depth PNG (uint16 or uint8)
    color_path: path to RGB PNG or JPG
    depth_scale: multiply depth values (e.g. RealSense depth is in millimeters)

    Returns:
        depth_frame: numpy array HxW (float32 depth in meters)
        color_image: numpy array HxWx3 (uint8 RGB)
    """

    # Load depth (16-bit PNG recommended)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError(f"Could not load depth image: {depth_path}")

    # Convert depth to meters if needed
    depth_frame = depth_img.astype(np.float32) * depth_scale

    # Load color image (BGR → RGB)
    color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Could not load color image: {color_path}")

    color_image = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    return depth_frame, color_image


def capture_from_index(idx, folder="captured_dataset", depth_scale=1.0):
    depth_path = f"{folder}/depth_{idx:03d}.png"
    color_path = f"{folder}/rgb_{idx:03d}.png"
    return capture_from_file(depth_path, color_path, depth_scale)


