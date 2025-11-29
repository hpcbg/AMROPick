from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='system_skill_pick',
            executable='system_skill_pick',
            name='system_skill_pick',
            output='screen',
            emulate_tty=True
        ),
        Node(
            package='object_detector',
            executable='object_detector',
            name='object_detector',
            output='screen'
        ),
        Node(
            package='robot_control',
            executable='robot_control',
            name='robot_control',
            output='screen'
        ),
    ])
