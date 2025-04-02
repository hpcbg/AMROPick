from setuptools import setup, find_packages

setup(
    name="amropick",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "torch",
        "ultralytics"
    ],
    description="AMROPick Python package for object detection.",
)
