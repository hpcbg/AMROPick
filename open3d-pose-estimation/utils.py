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


def capture_filtered(pipeline, align):
    # Filters
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)  # <-- keep depth resolution unchanged

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    colorizer = rs.colorizer()

    print("[INFO] Warming up sensor...")
    for _ in range(30):
        pipeline.wait_for_frames()

    # Capture and align
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    accumulated_depth = None
    num_frames = 10

    for _ in range(num_frames):
        depth = aligned_frames.get_depth_frame()

        # Apply filters
        depth = decimation.process(depth)
        depth = depth_to_disparity.process(depth)
        depth = spatial.process(depth)
        depth = temporal.process(depth)
        depth = disparity_to_depth.process(depth)
        depth = hole_filling.process(depth)

        # Convert to numpy
        depth_np = np.asanyarray(depth.get_data()).astype(np.float32)

        if accumulated_depth is None:
            accumulated_depth = depth_np
        else:
            accumulated_depth += depth_np

    # Average
    averaged_depth = (accumulated_depth / num_frames).astype(np.uint16)

    # colorized_depth = np.asanyarray(colorizer.colorize(depth).get_data())
    normalized = cv2.normalize(averaged_depth, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    colorized_depth = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    # colorized_depth = np.asanyarray(colorizer.colorize(averaged_depth).get_data())

    # return np.asanyarray(depth.get_data()), np.asanyarray(color_frame.get_data()), colorized_depth
    return np.asanyarray(averaged_depth), np.asanyarray(color_frame.get_data()), colorized_depth

