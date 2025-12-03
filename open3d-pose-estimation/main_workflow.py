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
    create_grasp_frame
)
from realsense_setup import start_realsense, capture_frames
from run_icp_alignment import run_alignment

def main():
    config = load_config()
    intermediate_results = config["paths"]["intermediate_results"]

    pipeline, align, profile = start_realsense()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_profile.get_intrinsics()

    print("[INFO] Capturing frame...")
    depth_frame, color_image, depth_vis = capture_frames(pipeline, align, profile)
    os.makedirs(intermediate_results, exist_ok=True)
    cv2.imwrite(os.path.join(intermediate_results, "captured_rgb.png"), color_image)
    cv2.imwrite(os.path.join(intermediate_results, "filtered_depth.png"), depth_vis)
    cv2.imwrite(os.path.join(intermediate_results, "depth_frame.png"), depth_frame)

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

    T_model_to_object = alignment.transformation
    T_model_to_robot = T_model_to_object

    print("[INFO] Transformation from model to robot base:")
    print(T_model_to_robot)

    model_pcd = o3d.io.read_point_cloud(model_path)
    model_pcd.transform(T_model_to_robot)

    full_pcd = create_full_pointcloud_from_rgbd(depth_frame, color_image, color_intr)

    o3d.visualization.draw_geometries(
        [full_pcd, model_pcd] + draw_frames(get_camera_pose(config), get_robot_pose(config))
        # [full_pcd, model_pcd]
    )

    # Example: Select grasp point on aligned model
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

    pipeline.stop()

if __name__ == "__main__":
    main()
