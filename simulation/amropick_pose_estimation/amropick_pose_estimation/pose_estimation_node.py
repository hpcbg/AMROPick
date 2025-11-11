#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge

import cv2
import numpy as np
import open3d as o3d
import yaml
import os
from ultralytics import YOLO
import joblib 

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
        color_topic = "/camera/camera/color/image_raw"
        # color_topic = "/room_camera/rgb"
        depth_topic = "/camera/camera/depth/image_rect_raw"
        info_topic = "/camera/camera/color/camera_info"
        
        # Setup subscribers using message_filters to synchronize topics
        self.get_logger().info('Setting up subscribers...')
        self.color_sub = message_filters.Subscriber(self, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        
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

        
        # ... (Step 1: Convert ROS Messages is the same) ...
        color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        camera_info = info_msg 
        intermediate_results = self.config["paths"]["intermediate_results"]
        os.makedirs(intermediate_results, exist_ok=True)
        cv2.imwrite(os.path.join(intermediate_results, "sim_captured_rgb.png"), color_image)

        # --- 2. Run Segmentation (YOLO) ---
        self.get_logger().info("[INFO] Running segmentation (Stage 1: YOLO)...")
        # 'masks' are the raw masks from YOLO
        # 'detections' from this model are "dumb" (e.g., [0, "plate"])
        masks, _, _ = run_segmentation(
            self.config["paths"]["model_weights_path"],
            color_image,
            self.config["valid_classes"], # This might just be ["plate"]
            confidence=self.config["segmentation"]["confidence_threshold"]
        )
        
        if len(masks) == 0:
            self.get_logger().error("[ERROR] No objects detected by YOLO.")
            return

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
        T_model_to_object = alignment.transformation
        T_model_to_robot = T_model_to_object

        self.get_logger().info("[INFO] Transformation from model to robot base:")
        print(T_model_to_robot)

        model_pcd = o3d.io.read_point_cloud(model_path)
        model_pcd.transform(T_model_to_robot)

        full_pcd = create_full_pointcloud_from_rgbd(depth_image, color_image, camera_info)

        # Visualize aligned model
        o3d.visualization.draw_geometries(
            [full_pcd, model_pcd] + draw_frames(get_camera_pose(self.config), get_robot_pose(self.config))
        )

        # Select grasp point on original model
        grasp_pts, grasp_normals = select_grasp_point_from_model(model_path)
        
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