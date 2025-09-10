# detect_workflow.py

import time
import cv2

import os
from typing import Optional

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from utils import (
    load_config, get_camera_pose, get_robot_pose,
    capture_filtered, run_segmentation,
    extract_masked_pointcloud, create_full_pointcloud_from_rgbd  # noqa: F401 (full cloud optional)
)
from realsense_utils import setup_pipeline
from run_icp_alignment import run_alignment


# ---------- math helpers ----------
def _matrix_to_euler_xyz_degrees(R: np.ndarray):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def _pose_to_tx_ty_tz_rx_ry_rz(T: np.ndarray):
    R, t = T[:3, :3], T[:3, 3]
    Rx, Ry, Rz = _matrix_to_euler_xyz_degrees(R)
    return (float(t[0]), float(t[1]), float(t[2]), float(Rx), float(Ry), float(Rz))


def _render_overlay_png(geoms, out_png, w=1280, h=960):
    """Render Open3D geometries to PNG off-screen."""
    from open3d.visualization import rendering as r
    renderer = r.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])

    scene.scene.set_indirect_light_intensity(7000)
    scene.scene.set_sun_light([0.577, 0.577, -0.577], [1, 1, 1], 75000)
    scene.scene.enable_sun_light(True)

    mat = r.MaterialRecord()
    mat.shader = "defaultUnlit"

    for i, g in enumerate(geoms):
        name = f"g{i}"
        if isinstance(g, o3d.geometry.PointCloud):
            mat.point_size = 2.0
        scene.add_geometry(name, g, mat)

    # Fit camera to all points
    all_pts = []
    for g in geoms:
        if hasattr(g, "points"):
            all_pts.append(np.asarray(g.points))
    if all_pts:
        P = np.vstack(all_pts)
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(P))
        center = aabb.get_center()
        extent = max(aabb.get_extent().max(), 1e-3)
    else:
        center = np.zeros(3)
        extent = 1.0

    cam = r.Camera()
    cam.set_projection(60.0, w / h, 0.01, 100.0, r.Camera.FovType.Vertical)
    eye = center + np.array([0, 0, extent * 2.2])
    up = np.array([0, -1, 0])
    renderer.setup_camera(cam, center, eye, up)

    img = renderer.render_to_image()
    o3d.io.write_image(out_png, img)


# ---------- main entry ----------
def detect_by_part_id(
    part_id: int,
    model_fs_path: Optional[str] = None,
    grasp_xyz=(0.0, 0.0, 0.0),
    render_overlay: bool = False,   # default False to avoid OffscreenRenderer hangs
):
    t0 = time.time()
    cfg = load_config()
    out_dir = cfg["paths"]["intermediate_results"]
    os.makedirs(out_dir, exist_ok=True)

    # Resolve model path (map 1→Plate1 etc.)
    if model_fs_path and os.path.exists(model_fs_path):
        model_path = model_fs_path
    else:
        model_path = f"object_models/Plate{part_id}.ply"
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}

    print(f"[WEB] part_id={part_id}  model={model_path}", flush=True)

    pipeline = None
    try:
        # 1) Capture
        t = time.time()
        pipeline, align, profile = setup_pipeline()
        depth_frame, color_img, depth_vis = capture_filtered(pipeline, align)
        cv2.imwrite(os.path.join(out_dir, "captured_rgb.png"), color_img)
        cv2.imwrite(os.path.join(out_dir, "filtered_depth.png"), depth_vis)
        print(f"[WEB] capture: {(time.time()-t):.2f}s", flush=True)

        # 2) Segmentation
        t = time.time()
        masks, detections, names = run_segmentation(
            cfg["paths"]["model_weights_path"],
            color_img,
            cfg["valid_classes"],
            confidence=cfg["segmentation"]["confidence_threshold"],
        )
        print(f"[WEB] segmentation: {(time.time()-t):.2f}s; detections={len(detections)}", flush=True)
        if not detections:
            return {"error": "No detections."}

        # Prefer label "Plate {part_id}", else first detection
        wanted_label = f"Plate {part_id}"
        chosen = 0
        for i, (_, label) in enumerate(detections):
            if str(label).strip() == wanted_label:
                chosen = i; break
        obj_idx = detections[chosen][0]
        mask = masks[obj_idx]
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"filtered_mask_{obj_idx}.png"), mask)

        # 3) Point cloud
        t = time.time()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        points, colors = extract_masked_pointcloud(mask, depth_frame, color_img, intr)
        cut_pcd = o3d.geometry.PointCloud()
        cut_pcd.points = o3d.utility.Vector3dVector(points)
        cut_pcd.colors = o3d.utility.Vector3dVector(colors)
        cut_scene_path = os.path.join(out_dir, "cut_scene.ply")
        o3d.io.write_point_cloud(cut_scene_path, cut_pcd)
        print(f"[WEB] masked cloud: {(time.time()-t):.2f}s  -> {cut_scene_path}", flush=True)

        # 4) ICP alignment
        t = time.time()
        alignment = run_alignment(model_path=model_path, scene_path=cut_scene_path)
        T_model_to_camera = alignment.transformation
        print(f"[WEB] ICP: {(time.time()-t):.2f}s", flush=True)

        # 5) Robot frame
        T_wc = get_camera_pose(cfg)   # world←camera
        T_wr = get_robot_pose(cfg)    # world←robot
        T_model_to_robot = np.linalg.inv(T_wr) @ T_wc @ T_model_to_camera

        pose_camera = _pose_to_tx_ty_tz_rx_ry_rz(T_model_to_camera)
        pose_robot  = _pose_to_tx_ty_tz_rx_ry_rz(T_model_to_robot)

        # 6) Visualization (robust fallback)
        # Try a fast preview image so the route returns quickly.
        preview_path = os.path.join(out_dir, "detection_preview.png")
        try:
            # simple colored overlay of mask on the RGB image (always works)
            overlay = color_img.copy()
            if mask.ndim == 2:
                overlay_mask = (mask > 0).astype(np.uint8) * 255
                overlay[..., 1] = np.where(overlay_mask > 0, np.minimum(overlay[..., 1] + 80, 255), overlay[..., 1])
                overlay = cv2.addWeighted(color_img, 0.7, overlay, 0.3, 0)
            # stamp pose text
            cv2.putText(overlay, f"Cam (Tx,Ty,Tz,Rx,Ry,Rz) = {pose_camera}", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"Robot (Tx,Ty,Tz,Rx,Ry,Rz) = {pose_robot}", (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(preview_path, overlay)
        except Exception as e:
            print(f"[WEB] preview overlay failed: {e}", flush=True)
            preview_path = os.path.join(out_dir, "captured_rgb.png")

        # Optional heavy overlay with Open3D offscreen (disabled by default)
        if render_overlay:
            try:
                model_geom = o3d.io.read_point_cloud(model_path)
                model_geom.transform(T_model_to_robot)
                overlay_png = os.path.join(out_dir, f"overlay_plate{part_id}.png")
                _render_overlay_png([cut_pcd, model_geom], overlay_png)
                preview_path = overlay_png
            except Exception as e:
                print(f"[WEB] OffscreenRenderer failed; keeping lightweight preview. Reason: {e}", flush=True)

        print(f"[WEB] total: {(time.time()-t0):.2f}s. Returning.", flush=True)
        return {
            "pose_camera": pose_camera,
            "pose_robot":  pose_robot,
            "overlay_png": preview_path,  # always a PNG that exists
        }

    finally:
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception:
            pass