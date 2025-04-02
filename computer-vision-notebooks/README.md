# Notebooks for AI model training for custom dataset

This folder contains various notebooks which illustrate the model training and prediction.

You need a dataset which can be generated from the simulation by using [dataset_generation.wbt](../webots/object_recognition_project/worlds/dataset_generation.wbt).

The developed object detection method is validated in Webots simulation [vision_pick_and_place.wbt](../webots/object_recognition_project/worlds/vision_pick_and_place.wbt).

The simulation is predicting the objects position and orientation in the same way as [predict-yolov8-multiple-stages.ipynb](./predict-yolov8-multiple-stages.ipynb). After some more tests, this notebook should be considered for the creation of the stand-alone python package for the AMROPick Vision System.
