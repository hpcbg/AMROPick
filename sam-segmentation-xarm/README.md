# Geometric Plates Detection with Robotic Pick and Place

This repository contains the complete algortihm for specific plates pick and place operation.

The used robot is xArm7 and the plates segmentation is done with Segment Anything model. The plates are classified by their binary region properties like area and proportions. The camera which is used in the experiments is a single standard high definition RGB camera

The repository contains the code needed for camera position calibration, plates binary region properties calculation and for the execution of the pick and place operation. Also, a simple xArm7 emulator of the XArmAPI which can be used for testing purposes without real robot.

## Requirements

The code is tested on Python 3.12. The following packages are required:

- ultralytics
- opencv-contrib-python
- jupyter
- xarm-python-sdk
- torch
- torchaudio
- torchvision
- segment-anything

## Initial Setup

The camera needs to be set on a stationary position. The camera device ID or URL should be set in `config/pick_and_place_gui.json`. The IDs of markers of the fixture for plane homography calibration should also be set in `config/pick_and_place_gui.json`.

Then the fixture should be placed on a position within the camera view and the script `utils/update_markers_config.py` should be executed to detect the markers position and update them in the configuration file. Next, you need to specify the pose of the fixture in the frame of the workspace of the robot in the `workarea` part of the `config/pick_and_place_gui.json`. You should also set the `fence` properties as they will prevent robot collisions.

Next, the objects properties must be calculated. For this you need to place the new objects within the workarea. The you need to capture a frame from the camera with the script `utils/capture.py`. Then, you can use the Jupyter Notebook `utils/process_new_objects.ipynb` to analyse the captured frame. This notebook will generate all the masks which are found by the segmentation. You need to look at the masks for your specific objects and manually copy the details about their properties in the file `config/objects.json`. If the object is a new object you should create a new entry for the object. You also need to verify the buttons list in the `config/pick_and_place_gui.json`. You can change their order and remove specific objects from the buttons list.

You should also set the place pose in the `config/pick_and_place_gui.json`.

In the `config/xarm_pick_and_place.json` you need to set the IP address of the robot and whether to run the script in XArmAPI emulation mode.

You should use the `tpl` files in the `config` for reference and for creation of the initial config files since the real config files are not tracked by the repo.

## Running Pick and Place

For the execution of pick and place you need to run the `xarm_pick_and_place.py` together with the GUI in the python script `pick_and_place_gui.py`. When you click on the button for the speciic object the processing pipeline is executed and the robot will pick and place the detected object.

## Utilities

There are several helper files in the `utils` folder:

- `detect_simple.ipynb` demonstrates the object segmentation and detection in a Jyputer Notebook.
- `detect_with_coordinates.ipynb` demonstrates the object segmentation, detection and global frame coordinates calculation in a Jyputer Notebook.
- `find_markers.ipynb` helper notebook for getting the IDs of the of AruCo tags.
- `xarm_emulator.py` is the XArmAPI emulator.

## Known Limitations and Future Updates

Perspective transform and camera checkboard calibration should be added.
