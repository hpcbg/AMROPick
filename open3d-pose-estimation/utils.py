import numpy as np
import cv2
import pyrealsense2 as rs

def extract_masked_pointcloud(mask, depth_frame, color_image, color_intr):
    points = []
    colors = []
    for v in range(mask.shape[0]):
        for u in range(mask.shape[1]):
            if mask[v, u] > 0:
                depth = depth_frame.get_distance(u, v)
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
