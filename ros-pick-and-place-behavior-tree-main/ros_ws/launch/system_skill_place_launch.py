from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='system_skill_place',
            executable='system_skill_place',
            name='system_skill_place',
            output='screen',
            emulate_tty=True
        ),
        Node(
            package='robot_control',
            executable='robot_control',
            name='robot_control',
            output='screen'
        ),
    ])
