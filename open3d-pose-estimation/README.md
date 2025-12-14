# Pose estimation using Open3D
This project demonstrate pose estimation in 3D space using Open3D and data from depth camera.

## Deployment

Install Python3.12, pip and virtualenv
```
sudo apt-get install -y python3.12-venv python3.12-distutils ca-certificates
```

Clone this repo
```
git clone https://github.com/hpcbg/AMROPick
cd AMROPick
```

Create and activate a virtualenv
```
python3.12 -m venv venv
source venv/bin/activate
```

Install the required packages
```
pip install --upgrade pip
pip install -r open3d-pose-estimation/requirements.txt
```

Go to the specific folder
```
cd open3d-pose-estimation
```

Run the main workflow
```
python main_workflow.py
```


To convert STL file to PLY
```
python convert_stl_to_ply.py
```

Capture images from realsense
```
python capture_from_realsense.py
```

Run the web app
```
python app.py
```