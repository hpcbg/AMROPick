#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped # <--- Added

import cv2
import numpy as np
import open3d as o3d
import yaml
import os
from ultralytics import YOLO
import joblib 

from scipy.spatial.transform import Rotation as R # <--- Added for math

# Use package-relative imports
from amropick_pose_estimation.run_icp_alignment import run_alignment
from amropick_pose_estimation.utils import (
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
    clean_mask,         
    extract_features    
)

def rotate_around_camera_axis(object_pose_matrix, axis, degrees):
    """
    Rotates the object pose around the CAMERA'S axes (Optical axes),
    regardless of where the object is.
    
    Args:
        object_pose_matrix: The 4x4 pose of the object.
        axis: 'x', 'y', or 'z'.
        degrees: Angle in degrees.
    """
    # 1. Create the Rotation Matrix
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    rot_matrix = np.eye(4)
    
    if axis.lower() == 'x':   # Camera Right/Left tilt
        rot_matrix[:3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis.lower() == 'y': # Camera Up/Down tilt
        rot_matrix[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis.lower() == 'z': # Camera Roll
        rot_matrix[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # 2. Apply the rotation as a PRE-MULTIPLICATION
    # This treats the rotation as occurring "before" the object moved away from the camera.
    # Mathematically: New_Pose = Rotation * Old_Pose
    return rot_matrix @ object_pose_matrix

def rotate_pose(pose_matrix, degrees, axis='z', local=True):
    """
    Rotates a 4x4 pose matrix by a specific angle along a specific axis.

    Args:
        pose_matrix (np.ndarray): The 4x4 input pose.
        degrees (float): Rotation angle in degrees.
        axis (str): 'x', 'y', or 'z'.
        local (bool): 
            - If True (default): Rotates around the object's OWN axis.
              (Useful for spinning the object).
            - If False: Rotates around the REFERENCE (Camera/World) axis.
              (Useful for fixing axis mapping issues).

    Returns:
        np.ndarray: The new 4x4 pose matrix.
    """
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    
    # Create the rotation matrix based on the axis
    rot_matrix = np.eye(4)
    
    if axis.lower() == 'x':
        rot_matrix[:3, :3] = np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])
    elif axis.lower() == 'y':
        rot_matrix[:3, :3] = np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        rot_matrix[:3, :3] = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Apply the rotation
    if local:
        # Post-multiply: Pose * Rotation
        return pose_matrix @ rot_matrix
    else:
        # Pre-multiply: Rotation * Pose
        return rot_matrix @ pose_matrix

# --- Helper Function for Matrix to Pose ---
def convert_to_pose_msg(transformation_matrix, frame_id, timestamp):
    """
    Converts a 4x4 numpy transformation matrix to a ROS 2 PoseStamped message.
    """
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.stamp = timestamp

    # Translation
    pose_msg.pose.position.x = transformation_matrix[0, 3]
    pose_msg.pose.position.y = transformation_matrix[1, 3]
    pose_msg.pose.position.z = transformation_matrix[2, 3]

    # Rotation (Matrix -> Quaternion)
    try:
        r = R.from_matrix(transformation_matrix[:3, :3])
        quat = r.as_quat() # returns [x, y, z, w]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
    except ValueError:
        # Handle cases where matrix is not a proper rotation matrix
        pose_msg.pose.orientation.w = 1.0

    return pose_msg

def create_grasp_orientation(normal_vector):
    """
    Creates a 3x3 rotation matrix where the Z-axis aligns with the normal vector.
    This is a basic heuristic.
    """
    z_axis = normal_vector / np.linalg.norm(normal_vector)
    
    # Create an arbitrary X-axis (that isn't parallel to Z)
    if np.abs(z_axis[0]) < 0.9:
        temp_axis = np.array([1, 0, 0])
    else:
        temp_axis = np.array([0, 1, 0])
        
    y_axis = np.cross(z_axis, temp_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    x_axis = np.cross(y_axis, z_axis)
    
    # Construct rotation matrix [x_axis, y_axis, z_axis]
    rot_matrix = np.eye(3)
    rot_matrix[:3, 0] = x_axis
    rot_matrix[:3, 1] = y_axis
    rot_matrix[:3, 2] = z_axis
    return rot_matrix

# --- Main ROS 2 Node Class ---

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        
        # Load config and CV bridge
        # Pass 'self' (the node) to load_config so it can find the package
        self.config = load_config(self)
        if self.config is None:
            self.get_logger().error("Failed to load config. Shutting down.")
            return
        
        self.roi_bounds = {
            'min_x': 428, 
            'max_x': 850, 
            'min_y': 0,  
            'max_y': 720
        }
        
        self.bridge = CvBridge()
        self.data_received = False

        self.get_logger().info("Loading classifier models...")
        try:
            clf_paths = self.config['paths']['classifier']
            self.shape_classifier = joblib.load(clf_paths['model_path'])
            self.scaler = joblib.load(clf_paths['scaler_path'])
            self.label_encoder = joblib.load(clf_paths['encoder_path'])
            self.feature_columns = joblib.load(clf_paths['features_path'])
            self.get_logger().info("Classifier models loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load classifier models: {e}")
            return
        
        # Define topics from simulator
        color_topic = "/rgb" #"/camera/camera/color/image_raw"
        # color_topic = "/room_camera/rgb"
        depth_topic = "/depth" #"/camera/camera/depth/image_rect_raw"
        info_topic = "/camera_info" # "/camera/camera/color/camera_info"
        
        # Setup subscribers using message_filters to synchronize topics
        self.get_logger().info('Setting up subscribers...')
        self.color_sub = message_filters.Subscriber(self, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)

        # --- Publishers ---
        # Published on demand when a pose is calculated
        self.object_pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', 10)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, '/calculated_grasp_pose', 10)
        
        self.ts = message_filters.TimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], 10
        )
        self.ts.registerCallback(self.listener_callback)
        
        self.get_logger().info(f"Node started. Waiting for synchronized data on topics:")
        self.get_logger().info(f"  Color: {color_topic}")
        self.get_logger().info(f"  Depth: {depth_topic}")
        self.get_logger().info(f"  Info:  {info_topic}")

    def listener_callback(self, color_msg, depth_msg, info_msg):
        # This callback runs only once, replicating the original script's behavior
        if self.data_received:
            return
        self.data_received = True
        
        self.get_logger().info('First synchronized frame received! Starting workflow...')

        # Print dims of color image
        self.get_logger().info(f"Color Image - Width: {color_msg.width}, Height: {color_msg.height}")

        # Capture timestamp and frame_id for consistency in publishing later
        current_timestamp = color_msg.header.stamp
        # Important: The points are extracted in the camera frame (usually optical)
        reference_frame = color_msg.header.frame_id
        
        # ... (Step 1: Convert ROS Messages is the same) ...
        color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        camera_info = info_msg 
        intermediate_results = self.config["paths"]["intermediate_results"]
        os.makedirs(intermediate_results, exist_ok=True)
        cv2.imwrite(os.path.join(intermediate_results, "sim_captured_rgb.png"), color_image)

        # --- 2. Run Segmentation (YOLO) ---
        self.get_logger().info("[INFO] Running segmentation...")
        all_masks, _, _ = run_segmentation(
            self.config["paths"]["model_weights_path"],
            color_image,
            self.config["valid_classes"],
            confidence=self.config["segmentation"]["confidence_threshold"]
        )
        
        if len(all_masks) == 0:
            self.get_logger().error("[ERROR] No objects detected by YOLO.")
            return
        
        # --- NEW: Filter Masks by ROI ---
        masks = []
        self.get_logger().info(f"[INFO] Filtering {len(all_masks)} masks by ROI...")
        
        for m in all_masks:
            # Calculate Centroid of the mask
            M = cv2.moments(m)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Check if centroid is inside bounds
                if (self.roi_bounds['min_x'] < cX < self.roi_bounds['max_x'] and 
                    self.roi_bounds['min_y'] < cY < self.roi_bounds['max_y']):
                    masks.append(m)
                else:
                    self.get_logger().info(f"Skipping object at ({cX}, {cY}) - Outside ROI")
        
        if len(masks) == 0:
            self.get_logger().error("[ERROR] Objects detected, but all were outside the ROI.")
            return

        self.get_logger().info(f"[INFO] {len(masks)} objects within ROI. Proceeding to classification...")

        # --- 3. Run Classification (Random Forest) ---
        self.get_logger().info(f"[INFO] Classifying {len(masks)} detected masks (Stage 2: Classifier)...")
        classified_detections = [] # This will be our NEW list of detections
        classified_masks = []
        
        for i, mask in enumerate(masks):
            # Clean up the mask (e.g., keep largest component)
            cleaned_mask = clean_mask(mask)
            classified_masks.append(cleaned_mask)
            if cleaned_mask.sum() < 50: # Skip tiny masks
                continue
            
            # Extract geometric features
            features = extract_features(cleaned_mask)
            if features is None:
                continue
            
            # Use the loaded feature_columns to ensure correct order
            try:
                feature_values = np.array([features[name] for name in self.feature_columns]).reshape(1, -1)
            except KeyError as e:
                self.get_logger().warn(f"Feature {e} not found. Skipping mask.")
                continue

            # Scale features and predict
            scaled_features = self.scaler.transform(feature_values)
            probabilities = self.shape_classifier.predict_proba(scaled_features)[0]
            max_proba = np.max(probabilities)
            
            # Check if prediction is confident enough
            clf_threshold = self.config.get("classifier_confidence_threshold", 0.3)
            if max_proba >= clf_threshold:
                class_idx = np.argmax(probabilities)
                class_label = self.label_encoder.inverse_transform([class_idx])[0]
                
                # Convert part label from "px" to "Part x" where x is the number
                class_label = f"Plate {class_label[1:]}"

                # Store the *index* of the mask and its *new label*
                classified_detections.append( (i, class_label) )
                

        # --- 4. Visualization and User Choice ---
        if not classified_detections:
            self.get_logger().error("[ERROR] No objects passed the classifier threshold.")
            return
        else:
            self.get_logger().info(f"[INFO] {len(classified_detections)} objects classified successfully.")
            # How many masks kept
            self.get_logger().info(f"[INFO] {len(classified_masks)} masks kept after classification.")

        # Note: We pass the *new* classified masks and detections
        vis_image = visualize_detections(
            color_image, classified_masks, classified_detections, None,
            os.path.join(intermediate_results, "sim_detection_preview.png")
        )
        cv2.imshow("Detected Objects (Classified)", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.get_logger().info("Detected and classified objects:")
        for i, (idx, label) in enumerate(classified_detections):
            self.get_logger().info(f"{i}: {label} (from original mask {idx})")

        # --- 5. Continue with Pose Estimation ---
        choice = int(input("Choose object index: "))
        
        # 'detections' is now our classified list
        obj_mask_index = classified_detections[choice][0]
        class_label = classified_detections[choice][1]
        model_path = self.config["model_mapping"].get(class_label)
        
        if model_path is None:
            self.get_logger().error(f"[ERROR] No model defined for class {class_label}")
            return

        # Get the correct mask from the *original* YOLO masks list
        mask = masks[obj_mask_index]
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        self.get_logger().info("[INFO] Extracting masked point cloud...")
        points, colors = extract_masked_pointcloud(mask, depth_image, color_image, camera_info)
        cut_pcd = o3d.geometry.PointCloud()
        cut_pcd.points = o3d.utility.Vector3dVector(points)
        cut_pcd.colors = o3d.utility.Vector3dVector(colors)
        cut_scene_path = os.path.join(intermediate_results, "sim_cut_scene.ply")
        o3d.io.write_point_cloud(cut_scene_path, cut_pcd)

        self.get_logger().info("[INFO] Running alignment...")
        alignment = run_alignment(model_path=model_path, scene_path=cut_scene_path)

        self.get_logger().info("[INFO] Alignment complete. Transformation matrix:")
        print(alignment.transformation)
        # T_model_to_object = alignment.transformation
        # T_model_to_robot = T_model_to_object

        
        # This is the pose in the Optical Frame (Z is forward)
        T_model_to_robot = alignment.transformation 
        
        # --- Convert to ROS Frame ---
        # Example: Flip the whole coordinate system 180 degrees around X
        # T_model_to_robot = rotate_pose(T_model_to_robot, 180, 'x', local=False)
        T_model_to_robot = rotate_around_camera_axis(T_model_to_robot, 'y', 180)
        
        # T_ros_world = T_optical_to_ros * T_optical_pose
        # T_model_to_robot = T_optical_to_ros @ T_model_to_optical
        self.get_logger().info("[INFO] Transformation from model to camera:")
        print(T_model_to_robot)

        # --- 6. Publish Object Pose ---
        object_pose_msg = convert_to_pose_msg(T_model_to_robot, reference_frame, current_timestamp)
        # object_pose_msg.pose.position.z = 0.006  # Lift object pose by 2 cm for safety
        # Divide all position values by 10 to convert from cm to m
        object_pose_msg.pose.position.x *= 10.0
        object_pose_msg.pose.position.y *= 10.0
        object_pose_msg.pose.position.z = 2.0
        self.object_pose_pub.publish(object_pose_msg)
        self.get_logger().info(f"Published Object Pose to /detected_object_pose (Frame: {reference_frame})")

        model_pcd = o3d.io.read_point_cloud(model_path)
        model_pcd.transform(T_model_to_robot)

        full_pcd = create_full_pointcloud_from_rgbd(depth_image, color_image, camera_info)

        # Visualize aligned model
        o3d.visualization.draw_geometries(
            [full_pcd, model_pcd] + draw_frames(get_camera_pose(self.config), get_robot_pose(self.config))
        )

        # Select grasp point on original model
        grasp_pts, grasp_normals = select_grasp_point_from_model(model_path)

        # Take the first available grasp point (or add logic to select specific one)
        if len(grasp_pts) > 0:
            # Transform grasp point to world frame
            # p_world = R * p_model + t
            g_pt_world = (T_model_to_robot[:3, :3] @ grasp_pts[0] + T_model_to_robot[:3, 3])
            g_norm_world = (T_model_to_robot[:3, :3] @ grasp_normals[0])
            
            # Construct Grasp Matrix (Pos + Orientation based on Normal)
            grasp_rot_mat = create_grasp_orientation(g_norm_world)
            grasp_transform_mat = np.eye(4)
            grasp_transform_mat[:3, :3] = grasp_rot_mat
            grasp_transform_mat[:3, 3] = g_pt_world

            # Publish Grasp Pose
            grasp_pose_msg = convert_to_pose_msg(grasp_transform_mat, reference_frame, current_timestamp)
            self.grasp_pose_pub.publish(grasp_pose_msg)
            self.get_logger().info(f"Published Grasp Pose to /calculated_grasp_pose")
            
            # Create frames for visualization
            grasp_frames = [create_grasp_frame(g_pt_world, g_norm_world)]
        
        # Transform grasp points to robot frame
        grasp_pts_world = [(T_model_to_robot[:3, :3] @ p + T_model_to_robot[:3, 3]) for p in grasp_pts]
        grasp_normals_world = [(T_model_to_robot[:3, :3] @ n) for n in grasp_normals]
        grasp_frames = [create_grasp_frame(p, n) for p, n in zip(grasp_pts_world, grasp_normals_world)]

        # Visualize with grasp frames
        self.get_logger().info("Showing final visualization with grasp points...")
        o3d.visualization.draw_geometries(
            [full_pcd, model_pcd]
            + grasp_frames
            + draw_frames(get_camera_pose(self.config), get_robot_pose(self.config))
        )

        self.get_logger().info("Workflow complete. Shutting down.")
        self.destroy_node()

        

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()