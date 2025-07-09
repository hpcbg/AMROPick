import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os
import argparse


def generate_dummy_mask(image_shape, radius=100):
    h, w = image_shape[:2]
    center = (w // 2, h // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, color=255, thickness=-1)
    return mask


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

    if len(points) == 0:
        raise RuntimeError("No valid 3D points extracted from mask and depth.")
    return np.array(points), np.array(colors)


def main():
    parser = argparse.ArgumentParser(description="Capture and save RGBD data with optional mask cutting.")
    parser.add_argument('--cut', action='store_true', help="Enable mask-based cutting of the scene.")
    parser.add_argument('--save-full', action='store_true', help="Save full scene point cloud.")
    parser.add_argument('--save-cut', action='store_true', help="Save cut/masked point cloud (only if --cut is enabled).")
    parser.add_argument('--output', type=str, default="output/capture", help="Base output path (without extension)")
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    print("[INFO] Warming up sensor...")
    for _ in range(30):
        pipeline.wait_for_frames()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_profile.get_intrinsics()
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        width=color_intr.width, height=color_intr.height,
        fx=color_intr.fx, fy=color_intr.fy,
        cx=color_intr.ppx, cy=color_intr.ppy
    )

    os.makedirs("../../pose-estimation-open3d/output", exist_ok=True)
    counter = 0

    while True:
        print("[INFO] Press 's' to save, 'q' to quit...")
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if args.cut:
            # mask = generate_dummy_mask(color_image.shape, radius=100)
            mask = cv2.imread("output2/rgb_000.png_0.jpg", cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (1920, 1080))
            overlay = cv2.addWeighted(color_image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.imshow("RGB + Mask", overlay)
        else:
            cv2.imshow("RGB", color_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"[INFO] Saving data set #{counter}")

            # Save RGB image
            img_path = f"output2/rgb_{counter:03d}.png"
            cv2.imwrite(img_path, color_image)
            print(f"[INFO] Saved image: {img_path}")

            img_path = f"output2/depth_{counter:03d}.png"
            cv2.imwrite(img_path, depth_image)
            print(f"[INFO] Saved image: {img_path}")


            # Save full scene
            if args.save_full:
                color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                depth_o3d = o3d.geometry.Image(depth_image)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_scale=1000.0,
                    convert_rgb_to_intensity=False,
                    depth_trunc=3.0
                )
                full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
                full_pcd.transform([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
                full_path = f"output2/scene_full_{counter:03d}.ply"
                o3d.io.write_point_cloud(full_path, full_pcd)
                print(f"[INFO] Full scene saved to {full_path}")

            # Save cut scene
            if args.cut and args.save_cut:
                points, colors = extract_masked_pointcloud(mask, depth_frame, color_image, color_intr)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd.transform([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])
                cut_path = f"output2/scene_cut_{counter:03d}.ply"
                o3d.io.write_point_cloud(cut_path, pcd)
                print(f"[INFO] Cut scene saved to {cut_path}")

            counter += 1

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
