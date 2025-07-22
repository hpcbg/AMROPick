import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs

def extract_masked_pointcloud(mask, depth_frame, color_image, color_intr):
    points = []
    colors = []
    for v in range(mask.shape[0]):
        for u in range(mask.shape[1]):
            if mask[v, u] > 0:
                # depth = depth_frame.get_distance(u, v)
                depth = depth_frame[v, u] / 1000.0  # Convert mm to meters if needed
                if depth == 0:
                    continue
                xyz = rs.rs2_deproject_pixel_to_point(color_intr, [u, v], depth)
                points.append(xyz)
                colors.append(color_image[v, u] / 255.0)
    if not points:
        raise RuntimeError("No valid 3D points extracted.")
    return np.array(points), np.array(colors)


def generate_dummy_mask(image_shape, radius=100):
    h, w = image_shape[:2]
    center = (w // 2, h // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, color=255, thickness=-1)
    return mask


def capture_filtered(pipeline, align, n_frames=5, apply_average_filter=False):
    print("[INFO] Warming up sensor...")
    for _ in range(30):
        pipeline.wait_for_frames()

    # Capture and align
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Initialize RealSense filters
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    colorizer = rs.colorizer()

    if apply_average_filter:
        # Accumulators
        depth_accum = None
        color_accum = None

    for _ in range(n_frames):
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        
        # Apply filters to depth
        depth = depth_to_disparity.process(depth_frame)
        depth = spatial.process(depth)
        depth = temporal.process(depth)
        depth = disparity_to_depth.process(depth)
        depth = hole_filling.process(depth)

        if apply_average_filter:
            depth_np = np.asanyarray(depth.get_data()).astype(np.float32)

            if depth_accum is None:
                depth_accum = depth_np
                color_accum = color_image.astype(np.float32)
            else:
                depth_accum += depth_np
                color_accum += color_image.astype(np.float32)

    if apply_average_filter:
        # Average over frames
        averaged_depth = (depth_accum / n_frames).astype(np.uint16)
        averaged_color = (color_accum / n_frames).astype(np.uint8)

        # Colorized depth for visualization (optional)
        try:
            colorized_depth = np.asanyarray(colorizer.colorize(depth).get_data())
        except Exception:
            colorized_depth = np.zeros_like(averaged_color)

        colorized_depth = np.asanyarray(colorizer.colorize(depth).get_data())

        return averaged_depth, averaged_color, colorized_depth

    colorized_depth = np.asanyarray(colorizer.colorize(depth).get_data())    

    return np.asanyarray(depth.get_data()), np.asanyarray(color_frame.get_data()), colorized_depth

