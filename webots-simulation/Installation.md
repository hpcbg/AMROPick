# Installation

This document shows how you can install Anaconda, Webots and all the required packages in order to run the code from this repository.

## Installation for model training and inference on Windows 11 with NVIDIA CPU or GPU support

1. Install Anaconda: [https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe)

2. Open Anaconda Prompt

3. Create a virtual environment:

   `conda create -n amropick python=3.9`

4. Activate the environment (from Anaconda Prompt):

   `conda activate amropick`

5. Install Python modules

   ```
   pip install jupyter
   pip install opencv-python
   pip install matplotlib
   pip install ultralytics
   ```

   This will install the CPU version of PyTorch. It will be slow for training but OK for inference.
   You can verify it with the following command from Anaconda environment:
   `python` and then type in the interpreter:

   ```
   Python 3.9.21 (main, Dec 11 2024, 16:35:24) [MSC v.1929 64 bit (AMD64)] on win32
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import torch
   >>> torch.**version**
   >>> '2.6.0+cpu'
   ```

   You can also install all the packages with:

   `pip install -r requirements_CPU.txt`.

6. Install `amropick-python` to your local virtual environment:

   ```
   conda activate amropick
   cd amropick-vision
   pip install -e .
   ```

7. If you want to install the GPU version of PyTorch go to the PyTorch Get Started web site: https://pytorch.org/get-started/locally/ and select your environment. Get the command and add the `--upgrade` flag.
   The command will look like this:

   `pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

   Verify the GPU installation from `python` interpreter:

   ```
   Python 3.9.21 (main, Dec 11 2024, 16:35:24) [MSC v.1929 64 bit (AMD64)] on win32
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import torch
   >>> torch.**version**
   >>> '2.6.0+cu126'
   >>> torch.cuda.is_available()
   >>> True
   ```

8. Now, you can start Jupyter with:

   ```
   cd computer-vision-notebooks
   jupyter-lab
   ```

9. You can now execute the Jupyter Notebooks!

Further, if you want to execute the simulations you need to continue with the Webots installation.

## Webots installation

1. Install Webots (it is recommended to install it for all users): [https://cyberbotics.com/](https://cyberbotics.com/)

2. Install Python modules required for the simulation:

   ```
   pip install PyQt5
   pip install spatialmath-python
   pip install roboticstoolbox-python
   pip install --upgrade numpy==1.26.4
   ```

   Note: Downgrade of the NumPy module to version below 2 is required.

3. Webots requires Python, so you need to start it always from your virtual environment. Open Anaconda Prompt and then:

   ```
   conda activate amropick
   "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
   ```

4. Now you can open the world file for the simulation, e.g.: [vision_pick_and_place.wbt](./Webots/object_recognition_project/worlds/vision_pick_and_place.wbt)!
