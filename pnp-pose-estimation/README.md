# Find pose with Perspective-n-Point (PnP) algorithm

Find the object pose (translation and orientation) by using Perspective-n-Point (PnP) and single or stereo camera.

How to run it in virtual environment:

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install opencv-python`
4. `pip install open3d`
5. `pip install matplotlib`
6. `pip install jupyterlab`
7. `python find_pose.py`

For OAK-D camera:

1. `pip install depthai`

For Segment Anything:

1. `pip install segment-anything`
2. `pip install torch torchvision`
3. Download model from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
