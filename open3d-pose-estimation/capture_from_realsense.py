import os
import cv2
import numpy as np
import open3d as o3d
import argparse

from utils import extract_masked_pointcloud, capture_filtered, load_config
from realsense_utils import setup_pipeline
import pyrealsense2 as rs


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Capture and save RGBD data with optional mask cutting.")
    parser.add_argument('--cut', action='store_true')
    parser.add_argument('--save-full', action='store_true')
    parser.add_argument('--save-cut', action='store_true')
    parser.add_argument('--output', type=str, default="captured_dataset", help="Output directory")
    parser.add_argument('--mask-path', type=str, help="Path to mask image")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    pipeline, align, profile = setup_pipeline()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_profile.get_intrinsics()
    color_sensor = profile.get_device().first_color_sensor()
    color_sensor.set_option(rs.option.sharpness, config["camera"]["rgb"]["sharpness"])
    color_sensor.set_option(rs.option.contrast, config["camera"]["rgb"]["contrast"])
    color_sensor.set_option(rs.option.gamma, config["camera"]["rgb"]["gamma"])
    color_sensor.set_option(rs.option.saturation, config["camera"]["rgb"]["saturation"])

    pinhole = o3d.camera.PinholeCameraIntrinsic(
        color_intr.width, color_intr.height,
        color_intr.fx, color_intr.fy,
        color_intr.ppx, color_intr.ppy)

    counter = 0

    while True:
        print("[INFO] Press 's' to save, 'q' to quit...")

        depth_frame, color_image, depth_vis = capture_filtered(pipeline, align)

        if args.cut and args.mask_path:
            # mask = generate_dummy_mask(color_image.shape, radius=100)
            mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (1920, 1080))
            overlay = cv2.addWeighted(color_image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.imshow("RGB + Mask", overlay)
        else:
            cv2.imshow("RGB", color_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"[INFO] Saving dataset #{counter}")

            cv2.imwrite(f"{args.output}/rgb_{counter:03d}.png", color_image)
            cv2.imwrite(f"{args.output}/depth_{counter:03d}.png", depth_frame)
            cv2.imwrite(f"{args.output}/depth_vis_{counter:03d}.png", depth_vis)

            if args.save_full:
                color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                depth_o3d = o3d.geometry.Image(depth_frame)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False, depth_trunc=3.0)
                full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
                full_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                o3d.io.write_point_cloud(f"{args.output}/scene_full_{counter:03d}.ply", full_pcd)

            if args.cut and args.save_cut and args.mask_path:
                points, colors = extract_masked_pointcloud(mask, depth_frame, color_image, color_intr)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                o3d.io.write_point_cloud(f"{args.output}/scene_cut_{counter:03d}.ply", pcd)

            counter += 1

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
