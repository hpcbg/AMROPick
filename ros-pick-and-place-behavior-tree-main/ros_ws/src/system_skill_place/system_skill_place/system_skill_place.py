import json

import time

import rclpy
from rclpy.node import Node
import py_trees
from rclpy.action import ActionClient

from std_msgs.msg import String

from custom_interfaces.action import RotateGripper, MoveRobot, MoveGripper
from custom_interfaces.srv import Void


class ReportStatus(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='placing', access=py_trees.common.Access.READ
        )

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.status_publisher = self.node.create_publisher(
            String, '/system_skill_place/status_json', 1)

    def update(self):
        msg = String()
        msg.data = json.dumps({'active': self.blackboard.placing})
        self.status_publisher.publish(msg)
        return py_trees.common.Status.RUNNING


class PlaceActive(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='placing', access=py_trees.common.Access.READ)

    def update(self):
        if self.blackboard.placing:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING


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


class SetPlacePosition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None
        self.action_client = None

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.action_client = ActionClient(
            self.node, MoveRobot, '/robot_control/move')

    def update(self):
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.node.get_logger().error('Robot control unavailable!')
            return py_trees.common.Status.FAILURE
        goal_msg = MoveRobot.Goal()
        goal_msg.pose_json = json.dumps({'x': 1.4, 'y': 2.8, 'z': 0.6})
        self.node.get_logger().info('Send action to move to place position.')
        self.action_client.send_goal_async(goal_msg)
        return py_trees.common.Status.SUCCESS


class SetHomePosition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None
        self.action_client = None

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.action_client = ActionClient(
            self.node, MoveRobot, '/robot_control/move')

    def update(self):
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.node.get_logger().error('Robot control unavailable!')
            return py_trees.common.Status.FAILURE
        goal_msg = MoveRobot.Goal()
        goal_msg.pose_json = json.dumps({'x': 0.0, 'y': 0.0, 'z': 0.0})
        self.node.get_logger().info('Send action to move to home position.')
        self.action_client.send_goal_async(goal_msg)
        return py_trees.common.Status.SUCCESS


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
        self.node.get_logger().info('Send action to release gripper.')
        self.action_client.send_goal_async(goal_msg)
        return py_trees.common.Status.SUCCESS


class CompletePlacing(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='placing',
            access=py_trees.common.Access.WRITE
        )

    def update(self):
        self.blackboard.placing = False
        self.logger.info('Place completed.')
        return py_trees.common.Status.SUCCESS


class PlaceNode(Node):
    def __init__(self):
        super().__init__('system_skill_place')

        # behaviour
        root = py_trees.composites.Parallel(
            name='System Skill Place',
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(
                synchronise=False
            )
        )

        system_state = py_trees.composites.Sequence(
            'System State', memory=True)
        system_state.add_children([
            ReportStatus('Report Status'),
            RobotStatus('Robot Status'),
        ])

        go_to_place_position_seq = py_trees.composites.Sequence(
            'Go to Place Position', memory=True)
        go_to_place_position_seq.add_children([
            RobotAvailable('Robot Stationary?'),
            SetPlacePosition('Send Place Position'),
            WaitNode('Wait 1 sec', wait_time=1.0),
        ])

        execute_place_seq = py_trees.composites.Sequence(
            'Execute Place', memory=True)
        execute_place_seq.add_children([
            RobotAvailable("Robot at Place Position?"),
            ReleaseGripper('Release Object'),
            WaitNode('Wait 1 sec', wait_time=1.0),
            RobotAvailable('Robot Stationary?'),
        ])

        complete_placing_seq = py_trees.composites.Sequence(
            'Finish Place', memory=True)
        complete_placing_seq.add_children([
            SetHomePosition('Set Home Position'),
            WaitNode('Wait 1 sec', wait_time=1.0),
            RobotAvailable('Robot Stationary?'),
            CompletePlacing('Finish Place')
        ])

        place_seq = py_trees.composites.Sequence(
            'Place', memory=True)
        place_seq.add_children([
            PlaceActive('Active?'),
            go_to_place_position_seq,
            execute_place_seq,
            complete_placing_seq
        ])

        root.add_children([system_state, place_seq])

        # behaviour tree
        self.tree = py_trees.trees.BehaviourTree(root)
        self.tree.setup(node=self)
        # render the tree
        py_trees.display.render_dot_tree(
            root,
            name='system_skill_place',
            target_directory='diagrams'
        )
        py_trees.display.render_dot_tree(
            root,
            name='system_skill_place_with_bb',
            target_directory='diagrams',
            with_blackboard_variables=True
        )

        self.blackboard = py_trees.blackboard.Client(name='Global')
        self.blackboard.register_key(
            key='placing',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.placing = False

        # tick_tree timer
        self.timer = self.create_timer(1.0, self.tick_tree)

        self._start_placing_srv = self.create_service(
            Void, '/system_skill_place/execute', self.start_placing_callback)

        self._abort_placing_srv = self.create_service(
            Void, '/system_skill_place/abort', self.abort_placing_callback)

    def start_placing_callback(self, request, response):
        self.blackboard.placing = True

        status = {
            'status': 'OK'
        }
        response.status = json.dumps(status)

        self.get_logger().info('Place started.')

        return response

    def abort_placing_callback(self, request, response):
        self.blackboard.placing = False

        status = {
            'status': 'OK'
        }
        response.status = json.dumps(status)

        self.get_logger().info('place aborted.')

        return response

    def tick_tree(self):
        self.tree.tick()


def main(args=None):
    rclpy.init(args=args)
    node = PlaceNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
