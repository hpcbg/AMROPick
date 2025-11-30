import os
from ultralytics import YOLO

import matplotlib.pyplot as plt

import cv2

os.environ['WANDB_MODE'] = 'disabled'

# Load a model
target_model_n = YOLO('yolov8s-seg.pt')  # load a pretrained model
target_model_n.train(data="/home/carlos/Documents/S1S2/Simeon/datas/yolo_datasets/yolo_metal_dataset_sam2_ori_plus_2025-06-17-simeon/data.yaml", epochs=1000, device=0) 
                           
