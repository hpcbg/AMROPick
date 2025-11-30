import cv2
import depthai as dai
import numpy as np
import json
import os
from datetime import datetime
import argparse


class OAKDLiveCapture:
    def __init__(self, output_dir="oakd_captured_depths"):
        self.output_dir = output_dir
        self.pipeline = None
        self.device = None
        self.q_rgb = None
        self.q_depth = None
        self.calibration_data = None
        self.intrinsics = None
        self.capture_count = 0

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def setup_pipeline(self):
        """Setup DepthAI pipeline for live preview and capture"""
        pipeline = dai.Pipeline()

        # RGB Camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 480)  # Preview size for live view
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Set manual focus (adjust as needed)
        cam_rgb.initialControl.setManualFocus(130)

        # Depth cameras
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_left.setCamera("left")
        mono_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_right.setCamera("right")

        # Stereo depth configuration
        depth.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.DEFAULT)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(False)

        # Link stereo cameras to depth
        mono_left.out.link(depth.left)
        mono_right.out.link(depth.right)

        # Outputs
        rgb_out = pipeline.create(dai.node.XLinkOut)
        depth_out = pipeline.create(dai.node.XLinkOut)
        rgb_4k_out = pipeline.create(dai.node.XLinkOut)

        rgb_out.setStreamName("rgb_preview")
        depth_out.setStreamName("depth")
        rgb_4k_out.setStreamName("rgb_4k")

        # Link outputs
        cam_rgb.preview.link(rgb_out.input)  # Preview for live view
        cam_rgb.isp.link(rgb_4k_out.input)   # 4K for capture
        depth.depth.link(depth_out.input)

        self.pipeline = pipeline

    def connect_device(self):
        """Connect to OAK-D device"""
        try:
            self.device = dai.Device(self.pipeline)
            self.q_rgb = self.device.getOutputQueue(
                name="rgb_preview", maxSize=4, blocking=False)
            self.q_rgb_4k = self.device.getOutputQueue(
                name="rgb_4k", maxSize=4, blocking=False)
            self.q_depth = self.device.getOutputQueue(
                name="depth", maxSize=4, blocking=False)

            # Try to get calibration data
            try:
                self.calibration_data = self.device.readCalibrationData()
            except AttributeError:
                try:
                    self.calibration_data = self.device.getCalibraionData()
                except AttributeError:
                    try:
                        self.calibration_data = self.device.getCalibrationData()
                    except AttributeError:
                        print(
                            "Warning: Cannot access calibration data, using defaults")
                        self.calibration_data = None

            # Get intrinsics for 4K resolution
            if self.calibration_data is not None:
                try:
                    for socket in [dai.CameraBoardSocket.RGB, dai.CameraBoardSocket.CAM_A]:
                        try:
                            self.intrinsics = self.calibration_data.getCameraIntrinsics(
                                socket, 3840, 2160)
                            break
                        except:
                            continue
                    else:
                        raise Exception("Could not get intrinsics")
                except:
                    self.calibration_data = None

            if self.calibration_data is None:
                # Default intrinsics for OAK-D 4K
                self.intrinsics = np.array([
                    [2880.0, 0.0, 1920.0],
                    [0.0, 2880.0, 1080.0],
                    [0.0, 0.0, 1.0]
                ])
                print("Using default camera intrinsics")

            print("Camera connected successfully!")
            return True

        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return False

    def save_frame_data(self, rgb_frame, depth_frame):
        """Save RGB frame, depth frame, and camera metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_count += 1

        # Create capture directory
        capture_dir = os.path.join(
            self.output_dir, f"capture_{timestamp}_{self.capture_count:03d}")
        os.makedirs(capture_dir, exist_ok=True)

        # Save RGB frame (4K)
        rgb_path = os.path.join(capture_dir, "rgb_4k.jpg")
        cv2.imwrite(rgb_path, rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save depth frame
        depth_path = os.path.join(capture_dir, "depth.png")
        cv2.imwrite(depth_path, depth_frame.astype(np.uint16))

        # Save depth as numpy array for precise values
        depth_npy_path = os.path.join(capture_dir, "depth.npy")
        np.save(depth_npy_path, depth_frame)

        # Prepare camera metadata
        metadata = {
            "timestamp": timestamp,
            "capture_number": self.capture_count,
            "rgb_resolution": rgb_frame.shape,
            "depth_resolution": depth_frame.shape,
            "intrinsics_matrix": self.intrinsics.tolist(),
            "camera_info": {
                "fx": float(self.intrinsics[0, 0]),
                "fy": float(self.intrinsics[1, 1]),
                "cx": float(self.intrinsics[0, 2]),
                "cy": float(self.intrinsics[1, 2])
            },
            "files": {
                "rgb": "rgb_4k.jpg",
                "depth_png": "depth.png",
                "depth_npy": "depth.npy"
            }
        }

        # Add calibration data if available
        if self.calibration_data is not None:
            try:
                # Try to get additional calibration info
                metadata["has_calibration_data"] = True

                # Try to get stereo baseline
                try:
                    extrinsics = None
                    for socket_pair in [
                        (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C),
                        (dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT)
                    ]:
                        try:
                            extrinsics = self.calibration_data.getCameraExtrinsics(
                                socket_pair[0], socket_pair[1])
                            break
                        except:
                            continue

                    if extrinsics is not None:
                        baseline = np.linalg.norm(extrinsics[0])
                        metadata["stereo_baseline_mm"] = float(baseline)

                except Exception as e:
                    print(f"Could not get stereo baseline: {e}")

            except Exception as e:
                print(f"Error getting additional calibration data: {e}")
        else:
            metadata["has_calibration_data"] = False
            metadata["note"] = "Using default intrinsics - may be less accurate"

        # Save metadata
        metadata_path = os.path.join(capture_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Captured frame {self.capture_count} saved to: {capture_dir}")
        print(f"  RGB: {rgb_frame.shape} -> {rgb_path}")
        print(f"  Depth: {depth_frame.shape} -> {depth_path}")
        print(f"  Metadata: {metadata_path}")

        return capture_dir

    def run_live_preview(self):
        """Run live preview with capture functionality"""
        if not self.connect_device():
            return

        print("\n=== LIVE PREVIEW MODE ===")
        print("Press 's' to save current frame")
        print("Press 'q' to quit")
        print("Press 'f' to adjust focus (will prompt for value)")
        print("\nStarting live preview...")

        try:
            while True:
                # Get preview frame
                in_rgb = self.q_rgb.get()
                in_depth = self.q_depth.get()

                if in_rgb is not None:
                    preview_frame = in_rgb.getCvFrame()

                    # Add overlay information
                    overlay_frame = preview_frame.copy()
                    cv2.putText(overlay_frame, f"Captures: {self.capture_count}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(overlay_frame, "Press 's' to capture, 'q' to quit",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay_frame, "Press 'f' for focus adjustment",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow("OAK-D Live Preview", overlay_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Capture 4K frame
                    print("Capturing 4K frame...")

                    # Get 4K RGB frame
                    in_rgb_4k = self.q_rgb_4k.get()
                    in_depth_capture = self.q_depth.get()

                    if in_rgb_4k is not None and in_depth_capture is not None:
                        rgb_4k_frame = in_rgb_4k.getCvFrame()
                        depth_capture_frame = in_depth_capture.getFrame()

                        # Save the captured data
                        self.save_frame_data(rgb_4k_frame, depth_capture_frame)
                    else:
                        print("Failed to capture 4K frame")

                elif key == ord('f'):
                    # Focus adjustment
                    try:
                        focus_value = int(
                            input("\nEnter focus value (0-255, 0=closest, 255=infinity): "))
                        focus_value = max(0, min(255, focus_value))

                        # Note: Focus adjustment during runtime may not work with all DepthAI versions
                        print(
                            f"Focus set to {focus_value} (restart may be required for effect)")

                    except ValueError:
                        print("Invalid focus value")
                    except KeyboardInterrupt:
                        print("\nFocus adjustment cancelled")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            cv2.destroyAllWindows()
            if self.device:
                self.device.close()
                print("Camera disconnected")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='OAK-D Live Capture')
    parser.add_argument('--output-dir', default='oakd_captured_depths',
                        help='Output directory for captured frames')
    parser.add_argument('--focus', type=int, default=130,
                        help='Initial focus value (0-255)')

    args = parser.parse_args()

    # Create capture instance
    capture = OAKDLiveCapture(output_dir=args.output_dir)
    capture.setup_pipeline()

    # Run live preview
    capture.run_live_preview()

    print(f"\nTotal captures: {capture.capture_count}")
    if capture.capture_count > 0:
        print(f"Files saved in: {capture.output_dir}")
        print("Use the processing notebook to analyze captured data")


if __name__ == "__main__":
    main()
