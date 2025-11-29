import json

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from custom_interfaces.action import MoveRobot, MoveGripper, RotateGripper, EmergencyStopRobot

from std_msgs.msg import String


class RobotControl(Node):

    def __init__(self):
        super().__init__('robot_control')

        self.busy = False
        self.pose = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.target_pose = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.gripper = 1.0
        self.target_gripper = 1.0

        self.gripper_rotation = 0
        self.target_gripper_rotation = 0.0

        self.init_actions()

        self.pose_publisher = self.create_publisher(
            String, '/robot_control/pose_json', 1)
        self.gripper_publisher = self.create_publisher(
            String, '/robot_control/gripper_json', 1)
        self.status_publisher = self.create_publisher(
            String, '/robot_control/status_json', 1)
        self.timer = self.create_timer(1.0, self.main_loop)

    def main_loop(self):
        busy = False
        if self.target_pose['x'] + 0.1 < self.pose['x']:
            self.pose['x'] -= 0.1
            busy = True
        elif self.target_pose['x'] - 0.1 > self.pose['x']:
            self.pose['x'] += 0.1
            busy = True
        if self.target_pose['y'] + 0.1 < self.pose['y']:
            self.pose['y'] -= 0.1
            busy = True
        elif self.target_pose['y'] - 0.1 > self.pose['y']:
            self.pose['y'] += 0.1
            busy = True
        if self.target_pose['z'] + 0.1 < self.pose['z']:
            self.pose['z'] -= 0.1
            busy = True
        elif self.target_pose['z'] - 0.1 > self.pose['z']:
            self.pose['z'] += 0.1
            busy = True

        if self.target_gripper + 0.1 < self.gripper:
            self.gripper -= 0.1
            busy = True
        elif self.target_gripper - 0.1 > self.gripper:
            self.gripper += 0.1
            busy = True

        if self.target_gripper_rotation + 0.1 < self.gripper_rotation:
            self.gripper_rotation -= 0.1
            busy = True
        elif self.target_gripper_rotation - 0.1 > self.gripper_rotation:
            self.gripper_rotation += 0.1
            busy = True

        msg = String()
        msg.data = json.dumps(self.pose)
        self.pose_publisher.publish(msg)

        msg = String()
        msg.data = json.dumps(
            {'rotation': self.gripper_rotation, 'position': self.gripper})
        self.gripper_publisher.publish(msg)

        status = {
            'status': 'busy' if busy else 'available'
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_publisher.publish(msg)

    def execute_move_callback(self, goal_handle):
        self.target_pose = json.loads(goal_handle.request.pose_json)

        self.get_logger().info(
            f'Requested move {self.target_pose}.')

        result = MoveRobot.Result()
        result.success = True
        goal_handle.succeed()
        return result

    def execute_rotate_gripper_callback(self, goal_handle):
        self.target_gripper_rotation = json.loads(
            goal_handle.request.rotation_json)

        self.get_logger().info(
            f'Requested gripper rotation {self.target_gripper_rotation}.')

        result = RotateGripper.Result()
        result.success = True
        goal_handle.succeed()
        return result

    def execute_move_gripper_callback(self, goal_handle):
        self.target_gripper = json.loads(
            goal_handle.request.position_json)

        self.get_logger().info(
            f'Requested gripper position {self.target_gripper}.')

        result = MoveGripper.Result()
        result.success = True
        goal_handle.succeed()
        return result

    def execute_emergency_stop_callback(self, goal_handle):
        self.target_gripper = self.gripper
        self.target_gripper_rotation = self.gripper_rotation
        self.target_pose = self.pose

        self.get_logger().info(
            f'Requested emergency stop.')

        result = EmergencyStopRobot.Result()
        result.success = True
        goal_handle.succeed()
        return result

    def init_actions(self):
        self._move_action_server = ActionServer(
            self,
            MoveRobot,
            '/robot_control/move',
            self.execute_move_callback)
        self._rotate_gripper_action_server = ActionServer(
            self,
            RotateGripper,
            '/robot_control/rotate_gripper',
            self.execute_rotate_gripper_callback)
        self._move_gripper_action_server = ActionServer(
            self,
            MoveGripper,
            '/robot_control/move_gripper',
            self.execute_move_gripper_callback)
        self._move_gripper_action_server = ActionServer(
            self,
            EmergencyStopRobot,
            '/robot_control/emergency_stop',
            self.execute_emergency_stop_callback)


def main(args=None):
    rclpy.init(args=args)

    robot_control = RobotControl()

    rclpy.spin(robot_control)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    robot_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
