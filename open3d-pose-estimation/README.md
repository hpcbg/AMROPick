# Pose estimation using Open3D
This project demonstrate pose estimation in 3D space using Open3D and data from depth camera.

## Deployment

Install Python, pip, and virtualenv
```
sudo apt-get install python3
sudo apt install python3-pip
sudo apt install python3-virtualenv
```

Clone this repo
```
git clone git@github.com:stsvetanov/pose-estimation-open3d.git
cd pose-estimation-open3d
```

Create and activate a virtualenv
```
virtualenv venv
source venv/bin/activate
```

Install the required packages
```
pip install -r requirements.txt
```

Run pose estimation with global (RANSAC) and local (ICP) registration using systematic data
```
python run_icp_alignment.py --model icp_data/Plate4_topdown.ply  --scene icp_data/Plate4_topdown.ply --visualize  --init_rz 10 --init_x 0.17 --init_y 0.19  --init_z 0.005
```

Local registration on scene taking from Realsense 435 depth camera
```
python run_icp_alignment.py --model icp_data/Plate2_topdown.ply --scene icp_data/realsense_scene_2.ply --visualize --skip_ransac --voxel 0.0005 --init_rz 180 --init_x 0.12 --init_z 0.001 
```
Registration on cut scene
```
python run_icp_alignment.py --model icp_data/Plate2_topdown.ply  --scene output2/scene_cut_000.ply --visualize  --icp_threshold 0.05
```

To convert STL file to PLY
```
python convert_stl_to_ply.py --input data/3D-models/Plate2.stl --output icp_data/Plate2_topdown.ply --mode topdown --scene icp_data/realsense_scene_2.ply --z_noise 0.00002 --num_points 2000
```