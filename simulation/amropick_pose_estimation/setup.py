# setup.py

from setuptools import find_packages, setup
import os
from glob import glob  # <-- Add this import

package_name = 'amropick_pose_estimation'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- ADD THESE LINES TO INSTALL YOUR DATA ---
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.*')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*.*')),
        # --- END OF ADDED LINES ---
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Pose estimation node from YOLO, ICP, and Open3D.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimator = amropick_pose_estimation.pose_estimation_node:main',
        ],
    },
)