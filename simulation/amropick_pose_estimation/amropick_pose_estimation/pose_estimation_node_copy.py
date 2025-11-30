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
import torch

# --- NEW IMPORTS FOR SAM 2 ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# --- END NEW IMPORTS ---

# Use package-relative imports
from amropick_pose_estimation.run_icp_alignment import run_alignment
from amropick_pose_estimation.utils import (
    extract_masked_pointcloud,
    run_segmentation, # This is now replaced, but we keep utils from it
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
        self.config = load_config(self)
        if self.config is None:
            self.get_logger().error("Failed to load config. Shutting down.")
            return
        
        self.bridge = CvBridge()
        self.data_received = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # --- LOAD CLASSIFIER (Unchanged) ---
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
        
        # --- LOAD YOLOv8 and SAM 2 (NEW) ---
        self.get_logger().info("Loading YOLOv8 and SAM 2 models...")
        original_cwd = os.getcwd() # Store the original working directory
        try:
            # 1. Load YOLO (replaces run_segmentation)
            self.yolo_model = YOLO(self.config["paths"]["model_weights_path"])
            self.get_logger().info(f"YOLOv8 model loaded successfully from: {self.config['paths']['model_weights_path']}")
            
            # 2. Load SAM 2
            # Get paths from your config file
            sam_cfg_path = self.config['paths']['sam_cfg']
            sam_ckpt_path = self.config['paths']['sam_ckpt']

            # Correct logging
            self.get_logger().info(f"Loading SAM 2 config from: {sam_cfg_path}")
            self.get_logger().info(f"Loading SAM 2 weights from: {sam_ckpt_path}")

            # --- HYDRA/SAM 2 WORKAROUND ---
            # The build_sam2 function (using Hydra) cannot load an absolute
            # config path. We must temporarily change the working directory
            # to the config's directory and pass the filename.
            
            config_dir = os.path.dirname(sam_cfg_path)
            config_file = os.path.basename(sam_cfg_path)

            self.get_logger().info(f"Changing CWD to {config_dir} for SAM 2 config loading...")
            os.chdir(config_dir)

            # --- THIS IS THE FIX ---
            # Use the config *filename* (not the full path)
            # and the *variable* for the checkpoint path.
            self.sam2_model = build_sam2(config_file, 
                                         sam_ckpt_path,
                                         device=self.device)
            
            # Change CWD back to what it was
            os.chdir(original_cwd)
            self.get_logger().info(f"Restored CWD to {original_cwd}")
            # --- END WORKAROUND ---

            self.sam_predictor = SAM2ImagePredictor(self.sam2_model)
            self.get_logger().info("YOLOv8 and SAM 2 models loaded successfully.")

        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO/SAM 2 models: {e}")
            # Ensure CWD is restored even if an error occurs
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            return
        # --- END NEW MODEL LOADING ---
        
        # Define topics from simulator
        color_topic = "/camera/camera/color/image_raw"
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
        if self.data_received:
            return
        self.data_received = True
        
        self.get_logger().info('First synchronized frame received! Starting workflow...')

        # --- 1. Convert ROS Messages ---
        color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # SAM needs RGB
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        camera_info = info_msg 
        intermediate_results = self.config["paths"]["intermediate_results"]
        os.makedirs(intermediate_results, exist_ok=True)
        cv2.imwrite(os.path.join(intermediate_results, "sim_captured_rgb.png"), color_image)
        orig_h, orig_w, _ = color_image.shape

        # --- 2. Run Segmentation (YOLO + SAM 2) ---
        
        # --- Step 2A: YOLO Predictions (Get Bounding Boxes) ---
        self.get_logger().info("[INFO] Running segmentation (Stage 1: YOLO)...")
        yolo_results = self.yolo_model.predict(color_image, verbose=False)
        high_conf_boxes = []
        if yolo_results[0].masks is not None:
            for i, r in enumerate(yolo_results[0]):
                conf = r.boxes.conf[0]
                if conf >= self.config["segmentation"]["confidence_threshold"]:
                    high_conf_boxes.append(r.boxes.xyxy[0].cpu().numpy())
        
        if len(high_conf_boxes) == 0:
            self.get_logger().error("[ERROR] No objects detected by YOLO.")
            return

        # --- Step 2B: SAM 2 Refinement (Get High-Res Masks) ---
        self.get_logger().info(f"[INFO] Refining {len(high_conf_boxes)} masks (Stage 2: SAM 2)...")
        self.sam_predictor.set_image(color_image_rgb)
        sam2_raw_masks = []
        if high_conf_boxes:
            for input_box in high_conf_boxes:
                masks, scores, _ = self.sam_predictor.predict(box=input_box[None, :], multimask_output=True)
                # Keep the single best mask from SAM's output
                sam2_raw_masks.append(masks[np.argmax(scores)])
        
        # 'sam2_raw_masks' is now the new list of masks we will use.
        # This list replaces the old 'masks' from run_segmentation

        # --- 3. Run Classification (on SAM 2 Masks) ---
        self.get_logger().info(f"[INFO] Classifying {len(sam2_raw_masks)} SAM 2 masks (Stage 3: Classifier)...")
        classified_detections = [] 
        classified_masks_for_vis = [] # For visualization
        
        # This loop logic is identical, but iterates over sam2_raw_masks
        for i, mask in enumerate(sam2_raw_masks):
            # Clean up the mask (e.g., keep largest component) for feature extraction
            cleaned_mask = clean_mask(mask)
            classified_masks_for_vis.append(cleaned_mask) # Store for visualization
            
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
                # 'i' now refers to the index in sam2_raw_masks
                classified_detections.append( (i, class_label) )
                

        # --- 4. Visualization and User Choice ---
        if not classified_detections:
            self.get_logger().error("[ERROR] No objects passed the classifier threshold.")
            return
        else:
            self.get_logger().info(f"[INFO] {len(classified_detections)} objects classified successfully.")

        # Note: We pass the *new* cleaned SAM 2 masks and classified detections
        vis_image = visualize_detections(
            color_image, classified_masks_for_vis, classified_detections, None,
            os.path.join(intermediate_results, "sim_detection_preview.png")
        )
        cv2.imshow("Detected Objects (Classified)", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.get_logger().info("Detected and classified objects:")
        for i, (idx, label) in enumerate(classified_detections):
            self.get_logger().info(f"{i}: {label} (from SAM 2 mask {idx})")

        # --- 5. Continue with Pose Estimation ---
        choice = int(input("Choose object index: "))
        
        # 'detections' is now our classified list
        obj_mask_index = classified_detections[choice][0] # This is the index in sam2_raw_masks
        class_label = classified_detections[choice][1]
        model_path = self.config["model_mapping"].get(class_label)
        
        if model_path is None:
            self.get_logger().error(f"[ERROR] No model defined for class {class_label}")
            return

        # *** CRITICAL CHANGE ***
        # Get the correct mask from the *SAM 2 raw masks* list
        mask = sam2_raw_masks[obj_mask_index]
        # *** END CRITICAL CHANGE ***
        
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