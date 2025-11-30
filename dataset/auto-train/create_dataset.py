import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

import os
import sys
sys.path.append('gsam2/autodistill-grounded-sam-2')

import torch
import gc

from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot

from ultralytics import YOLO

import matplotlib.pyplot as plt

import cv2

def setup_model(box_threshold, text_threshold):
    """Sets up the ontology and loads the GroundedSAM2 model."""
    print("Setting up ontology and loading model...")
    # Define an ontology to map class names to our Grounded SAM 2 prompt
    ontology = CaptionOntology({
        "metal piece, metal part": "metal part",
    })

    # Load the base model
    base_model = GroundedSAM2(ontology=ontology, model="Grounding DINO",
                              grounding_dino_box_threshold=box_threshold,
                              grounding_dino_text_threshold=text_threshold)
    print("Model loaded successfully.")
    return base_model


model = setup_model(box_threshold=0.6, text_threshold=0.3)

# The main folder that contains your images and subfolders
INPUT_FOLDER = "datas/2025-06-17-simeon"

# The folder where images with valid detections will be moved
OUTPUT_FOLDER = "datas/datasets/2025-06-17-simeon"

dataset = model.label(INPUT_FOLDER, extension=".png", output_folder=OUTPUT_FOLDER)