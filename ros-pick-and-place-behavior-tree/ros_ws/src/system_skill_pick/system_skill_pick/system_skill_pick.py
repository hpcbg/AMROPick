import json

import time

import logging

import rclpy
from rclpy.node import Node
import py_trees
from rclpy.action import ActionClient

from std_msgs.msg import String

from custom_interfaces.action import MoveRobot, MoveGripper
from custom_interfaces.srv import Void


class ReportStatus(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='picking', access=py_trees.common.Access.READ
        )

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.status_publisher = self.node.create_publisher(
            String, '/system_skill_pick/status_json', 1)

    def update(self):
        msg = String()
        msg.data = json.dumps({'active': self.blackboard.picking})
        self.status_publisher.publish(msg)
        return py_trees.common.Status.RUNNING


class PickingActive(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='picking', access=py_trees.common.Access.READ)

    def update(self):
        if self.blackboard.picking:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING


class CheckObjectDetected(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.subscriber = None
        self.node = None
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='detected_object',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.detected_object = {'found': False}

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.subscriber = self.node.create_subscription(
            String, '/object_detector/detected_object_json', self.callback, 10
        )

    def callback(self, msg):
        self.blackboard.detected_object = json.loads(msg.data)

    def update(self):
        return py_trees.common.Status.RUNNING


class MoveToPickPose(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None
        self.action_client = None
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='detected_object', access=py_trees.common.Access.READ)

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.action_client = ActionClient(
            self.node, MoveRobot, '/robot_control/move')

    def update(self):
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.node.get_logger().error('Robot control unavailable!')
            return py_trees.common.Status.FAILURE

        goal_msg = MoveRobot.Goal()

        goal_msg.pose_json = json.dumps(
            self.blackboard.detected_object['pose'])
        self.node.get_logger().info(
            f'Send move action to the position of the detected object: {self.blackboard.detected_object['pose']}.')
        self.action_client.send_goal_async(goal_msg)
        return py_trees.common.Status.SUCCESS


class CloseGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None
        self.action_client = None

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.action_client = ActionClient(
            self.node, MoveGripper, '/robot_control/move_gripper')

    def update(self):
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.node.get_logger().error('Robot control unavailable!')
            return py_trees.common.Status.FAILURE
        goal_msg = MoveGripper.Goal()
        goal_msg.position_json = json.dumps(0.2)
        self.node.get_logger().info('Send action to close the gripper.')
        self.action_client.send_goal_async(goal_msg)
        return py_trees.common.Status.SUCCESS


class ReleaseGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None
        self.action_client = None

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.action_client = ActionClient(
            self.node, MoveGripper, '/robot_control/move_gripper')

    def update(self):
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.node.get_logger().error('Robot control unavailable!')
            return py_trees.common.Status.FAILURE
        goal_msg = MoveGripper.Goal()
        goal_msg.position_json = json.dumps(1.0)
        self.node.get_logger().info('Send action to open the gripper.')
        self.action_client.send_goal_async(goal_msg)
        return py_trees.common.Status.SUCCESS


class RobotStatus(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.pose_subscriber = None
        self.node = None

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='robot_status',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.robot_status = {'status': 'busy'}
        self.blackboard.register_key(
            key='robot_pose',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.robot_pose = {'x': 0, 'y': 0, 'z': 0}
        self.blackboard.register_key(
            key='robot_gripper',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.robot_gripper = {'rotation': 0.0, 'position': 1.0}

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.pose_subscriber = self.node.create_subscription(
            String, '/robot_control/pose_json', self.pose_callback, 10
        )
        self.gripper_subscriber = self.node.create_subscription(
            String, '/robot_control/gripper_json', self.gripper_callback, 10
        )
        self.status_subscriber = self.node.create_subscription(
            String, '/robot_control/status_json', self.status_callback, 10
        )

    def pose_callback(self, msg):
        self.blackboard.robot_pose = json.loads(msg.data)

    def gripper_callback(self, msg):
        self.blackboard.robot_gripper = json.loads(msg.data)

    def status_callback(self, msg):
        self.blackboard.robot_status = json.loads(msg.data)

    def update(self):
        return py_trees.common.Status.RUNNING


class RobotAvailable(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='robot_status', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='robot_pose', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='robot_gripper', access=py_trees.common.Access.READ)

    def update(self):
        robot_data = f'Pose ({self.blackboard.robot_pose['x']:.2f}, {self.blackboard.robot_pose['y']:.2f}, {self.blackboard.robot_pose['z']:.2f}), Gripper (position: {self.blackboard.robot_gripper['position']:.2f}, rotation: {self.blackboard.robot_gripper['rotation']:.2f})'
        if self.blackboard.robot_status['status'] == 'available':
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.info(
                f'Robot is moving... {robot_data}')
            return py_trees.common.Status.RUNNING


class ObjectAvailable(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='detected_object', access=py_trees.common.Access.READ)

    def update(self):
        if self.blackboard.detected_object['found']:
            self.logger.info(
                f'Object detected at position {self.blackboard.detected_object['pose']}')
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.info('Waiting for a object detection...')
            return py_trees.common.Status.RUNNING


class WaitNode(py_trees.behaviour.Behaviour):
    def __init__(self, name, wait_time: float):
        super().__init__(name)
        self.wait_time = wait_time
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()
        self.logger.info(f'Waiting for {self.wait_time:.1f} seconds...')

    def update(self):
        elapsed = time.time() - self.start_time
        if elapsed < self.wait_time:
            return py_trees.common.Status.RUNNING
        return py_trees.common.Status.SUCCESS


class CompletePicking(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='picking',
            access=py_trees.common.Access.WRITE
        )

    def update(self):
        self.blackboard.picking = False
        self.logger.info('Pick completed.')
        return py_trees.common.Status.SUCCESS


class PickNode(Node):
    def __init__(self):
        super().__init__('system_skill_pick')

        # behaviour
        root = py_trees.composites.Parallel(
            name='System Skill Pick',
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(
                synchronise=False
            )
        )

        system_state = py_trees.composites.Sequence(
            'System State', memory=True)
        system_state.add_children([
            ReportStatus('Report Status'),
            RobotStatus('Robot Status'),
            CheckObjectDetected('Object Detector Status')
        ])

        move_to_pick_seq = py_trees.composites.Sequence(
            'Move to Pick Position', memory=True)
        move_to_pick_seq.add_children([
            ObjectAvailable('Object Detected?'),
            RobotAvailable('Robot Stationary?'),
            ReleaseGripper('Release Gripper'),
            WaitNode('Wait 1 sec', wait_time=1.0),
            RobotAvailable('Robot Stationary?'),
            MoveToPickPose('Send Pick Position'),
            WaitNode('Wait 1 sec', wait_time=1.0)
        ])

        pick_seq = py_trees.composites.Sequence(
            'Pick', memory=True)
        pick_seq.add_children(
            [RobotAvailable('Robot At Pick Position?'),
             CloseGripper('Close Gripper'),
             WaitNode('Wait 1 sec', wait_time=1.0)])

        complete_picking_seq = py_trees.composites.Sequence(
            "Finish Pick", memory=True)
        complete_picking_seq.add_children([
            RobotAvailable('Robot Stationary?'),
            CompletePicking('Finish Pick')
        ])

        picking_seq = py_trees.composites.Sequence(
            'Pick', memory=True)
        picking_seq.add_children([
            PickingActive('Active?'),
            move_to_pick_seq,
            pick_seq,
            complete_picking_seq
        ])

        root.add_children([system_state, picking_seq])

        # behaviour tree
        self.tree = py_trees.trees.BehaviourTree(root)
        self.tree.setup(node=self)
        # render the tree
        py_trees.display.render_dot_tree(
            root,
            name='system_skill_pick',
            target_directory='diagrams'
        )
        py_trees.display.render_dot_tree(
            root,
            name='system_skill_pick_with_bb',
            target_directory='diagrams',
            with_blackboard_variables=True
        )

        self.blackboard = py_trees.blackboard.Client(name='Global')
        self.blackboard.register_key(
            key='picking',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.picking = False

        # tick_tree timer
        self.timer = self.create_timer(1.0, self.tick_tree)

        self._start_picking_srv = self.create_service(
            Void, '/system_skill_pick/execute', self.start_picking_callback)

        self._abort_picking_srv = self.create_service(
            Void, '/system_skill_pick/abort', self.abort_picking_callback)

    def start_picking_callback(self, request, response):
        self.blackboard.picking = True

        status = {
            'status': 'OK'
        }
        response.status = json.dumps(status)

        self.get_logger().info('Pick started')

        return response

    def abort_picking_callback(self, request, response):
        self.blackboard.picking = False

        status = {
            'status': 'OK'
        }
        response.status = json.dumps(status)

        self.get_logger().info('Pick aborted')

        return response

    def tick_tree(self):
        self.tree.tick()


def main(args=None):
    rclpy.init(args=args)
    node = PickNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
