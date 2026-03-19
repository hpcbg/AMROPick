import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import os
from utils import (
    extract_masked_pointcloud,
    run_segmentation,
    visualize_detections,
    create_full_pointcloud_from_rgbd,
    load_config,
    get_camera_pose,
    get_robot_pose,
    draw_frames,
    select_grasp_point_from_model, 
    create_grasp_frame,
    draw_model_on_image,
    load_frames_from_file
)
from realsense_setup import start_realsense, capture_frames
from run_icp_alignment import run_alignment

def main():
    config = load_config()
    intermediate_results = config["paths"]["intermediate_results"]

    print("[INFO] Capturing frame...")

    if config["source"] == "camera":
        print("[INFO] Capturing frame from RealSense...")
        pipeline, align, profile = start_realsense()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()
        depth_frame, color_image = capture_frames(pipeline, align, profile)
        pipeline.stop()
    elif config["source"] == "file":
        print(f"[INFO] Loading dataset #{config['file_index']} from file...")
        depth_frame, color_image, color_intr = load_frames_from_file(config["file_index"])

    print("[INFO] Running segmentation...")
    masks, detections, names = run_segmentation(
        config["paths"]["model_weights_path"],
        color_image,
        config["valid_classes"],
        confidence=config["segmentation"]["confidence_threshold"]
    )

    if not detections:
        print("[ERROR] No valid objects detected.")
        return

    vis_image = visualize_detections(
        color_image, masks, detections, names,
        os.path.join(intermediate_results, "detection_preview.png")
    )
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

    mask = masks[obj_idx]
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(intermediate_results, f"filtered_mask_{obj_idx}.png"), mask)

    points, colors = extract_masked_pointcloud(mask, depth_frame, color_image, color_intr)
    cut_pcd = o3d.geometry.PointCloud()
    cut_pcd.points = o3d.utility.Vector3dVector(points)
    cut_pcd.colors = o3d.utility.Vector3dVector(colors)
    cut_scene_path = os.path.join(intermediate_results, "cut_scene.ply")
    o3d.io.write_point_cloud(cut_scene_path, cut_pcd)

    print("[INFO] Alignment:")
    alignment = run_alignment(model_path=model_path, scene_path=cut_scene_path)

    # ICP output: Model → Camera
    T_model_to_camera = alignment.transformation
    # Camera → Robot (from config)
    T_camera_to_robot = get_camera_pose(config)
    # Final: Model → Robot
    T_model_to_robot = T_camera_to_robot @ T_model_to_camera

    model_pcd = o3d.io.read_point_cloud(model_path)
    model_pcd.transform(T_model_to_robot)

    full_pcd = create_full_pointcloud_from_rgbd(depth_frame, color_image, color_intr)
    o3d.visualization.draw_geometries(
        [full_pcd, model_pcd] + draw_frames(get_camera_pose(config), get_robot_pose(config))
    )

    # Select grasp point on aligned model
    grasp_pts, grasp_normals = select_grasp_point_from_model(model_path)
    # grasp_frames = [create_grasp_frame(p, n) for p, n in zip(grasp_pts, grasp_normals)]

    # Transform grasp points to robot frame
    grasp_pts_world = [(T_model_to_robot[:3, :3] @ p + T_model_to_robot[:3, 3]) for p in grasp_pts]
    grasp_normals_world = [(T_model_to_robot[:3, :3] @ n) for n in grasp_normals]
    grasp_frames = [create_grasp_frame(p, n) for p, n in zip(grasp_pts_world, grasp_normals_world)]

    # Visualize with grasp frames
    o3d.visualization.draw_geometries(
        [full_pcd, model_pcd]
        + grasp_frames
        + draw_frames(get_camera_pose(config), get_robot_pose(config))
    )

    print("[INFO] Drawing CAD model overlay on the captured image...")
    # TODO Drawing lines from STL file works with limited models. 
    # class_id = class_label.split()[1]
    # model_mesh = o3d.io.read_triangle_mesh(f"object_models/Plate{class_id}.stl")
    # model_mesh.scale(0.001, center=(0,0,0))

    model_mesh = o3d.io.read_triangle_mesh(model_path)

    overlay = draw_model_on_image(
        image=color_image,
        mesh=model_mesh,                     # original mesh (not robot-transformed)
        T_model_cam=T_model_to_camera,  # ICP gives model→camera
        intrinsics=color_intr
    )

    cv2.imshow("CAD Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
