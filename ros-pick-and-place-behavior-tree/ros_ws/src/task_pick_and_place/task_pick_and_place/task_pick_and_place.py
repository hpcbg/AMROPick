import json

import time

import rclpy
from rclpy.node import Node
import py_trees

from std_msgs.msg import String

from custom_interfaces.srv import Void

from enum import IntEnum


class Stages(IntEnum):
    ACTIVE = 0
    PICK_EXECUTE = 1
    PICK_READY = 2
    PLACE_EXECUTE = 3
    PLACE_READY = 4
    ABORT = 10


class ReportStatus(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='active', access=py_trees.common.Access.READ
        )

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.status_publisher = self.node.create_publisher(
            String, '/task_pick_and_place/status_json', 1)

    def update(self):
        msg = String()
        msg.data = json.dumps({'active': self.blackboard.active})
        self.status_publisher.publish(msg)
        return py_trees.common.Status.RUNNING


class PickAndPlaceActive(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='active', access=py_trees.common.Access.READ)

    def update(self):
        if self.blackboard.active:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class SystemCheck(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='system_status', access=py_trees.common.Access.READ)

    def update(self):
        if self.blackboard.system_status == 'ok':
            self.logger.info('System is OK.')
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.info('System Error!')
            return py_trees.common.Status.FAILURE


class StageCheck(py_trees.behaviour.Behaviour):
    def __init__(self, name, stage):
        super().__init__(name)
        self.stage = stage
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='stage', access=py_trees.common.Access.READ)

    def update(self):
        if self.blackboard.stage == self.stage:
            self.logger.info('OK')
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class IdleStatus(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='stage', access=py_trees.common.Access.READ)

    def update(self):
        self.logger.info(f'Stage {self.blackboard.stage.name} running...')
        return py_trees.common.Status.SUCCESS


class SystemMonitorStatus(py_trees.behaviour.Behaviour):
    def __init__(self, name, ):
        super().__init__(name)
        self.subscriber = None
        self.node = None
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='system_status',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.system_status = 'unknown'

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.subscriber = self.node.create_subscription(
            String, '/system_monitor/status_json', self.callback, 10
        )

    def callback(self, msg):
        data = json.loads(msg.data)
        self.blackboard.system_status = data

    def update(self):
        return py_trees.common.Status.RUNNING


class SystemSkillsStatus(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.node = None

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='stage',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key='pick_status',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.pick_status = {'status': 'unknown'}

        self.blackboard.register_key(
            key='place_status',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.place_status = {'status': 'unknown'}

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.pick_subscriber = self.node.create_subscription(
            String, '/system_skill_pick/status_json', self.pick_callback, 10
        )
        self.place_subscriber = self.node.create_subscription(
            String, '/system_skill_place/status_json', self.place_callback, 10
        )

    def pick_callback(self, msg):
        self.blackboard.pick_status = json.loads(msg.data)
        if self.blackboard.pick_status['active'] == False and self.blackboard.stage == Stages.PICK_EXECUTE:
            self.blackboard.stage = Stages.PICK_READY

    def place_callback(self, msg):
        self.blackboard.place_status = json.loads(msg.data)
        if self.blackboard.place_status['active'] == False and self.blackboard.stage == Stages.PLACE_EXECUTE:
            self.blackboard.stage = Stages.PLACE_READY

    def update(self):
        return py_trees.common.Status.RUNNING


class CallService(py_trees.behaviour.Behaviour):
    def __init__(self, name, service, stage):
        super().__init__(name)
        self.node = None
        self.service = service
        self.stage = stage
        self.client = None
        self.future = None
        self.called = False

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='stage',
            access=py_trees.common.Access.WRITE
        )

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError:
            self.node = rclpy.create_node(f'service_caller_stage_{self.stage}')

        self.client = self.node.create_client(Void, self.service)

    def update(self):
        if self.stage == Stages.ABORT or self.blackboard.stage < self.stage:
            if not self.client.service_is_ready():
                self.logger.warning(f'Service {self.service} unavailable!')
                return py_trees.common.Status.FAILURE
            if self.future is None:
                request = Void.Request()
                self.future = self.client.call_async(request)
                return py_trees.common.Status.RUNNING
            if self.future.done():
                if self.future.result() is not None:
                    self.logger.info(f'Service {self.service} called.')
                    self.blackboard.stage = self.stage
                    self.future = None
                    return py_trees.common.Status.SUCCESS
                else:
                    self.logger.error(
                        f'Service {self.service} error: {self.future.exception()}')
                    return py_trees.common.Status.FAILURE
            self.logger.info(f'Service {self.service} running...')
            return py_trees.common.Status.RUNNING
        self.logger.info('Service failure!')
        return py_trees.common.Status.FAILURE


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


class CompletePickAndPlace(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key='active',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key='stage',
            access=py_trees.common.Access.WRITE
        )

    def update(self):
        self.blackboard.active = False
        self.logger.info('Pack Bottle completed.')
        return py_trees.common.Status.SUCCESS


class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('task_pick_and_place')

        # behaviour
        root = py_trees.composites.Parallel(
            name='Task Pick And Place',
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(
                synchronise=False
            )
        )

        system_state = py_trees.composites.Sequence(
            'System State', memory=True)
        system_state.add_children([
            ReportStatus('Report Status'),
            SystemSkillsStatus('System Skills Status'),
            SystemMonitorStatus('System Monitor Status'),
        ])

        pick_seq = py_trees.composites.Sequence(
            'Pick', memory=True)
        pick_seq.add_children([
            StageCheck('System Active?', Stages.ACTIVE),
            CallService('Call Pick', '/system_skill_pick/execute',
                        Stages.PICK_EXECUTE),
            WaitNode('Wait 1 sec', wait_time=1.0)
        ])

        place_seq = py_trees.composites.Sequence(
            'Place', memory=True)
        place_seq.add_children([
            StageCheck('Pick Ready?', Stages.PICK_READY),
            CallService(
                'Call Place', '/system_skill_place/execute', Stages.PLACE_EXECUTE),
            WaitNode('Wait 1 sec', wait_time=1.0),
        ])

        complete_seq = py_trees.composites.Sequence(
            'Finish Pick And Place', memory=True)
        complete_seq.add_children([
            StageCheck('Place Ready?', Stages.PLACE_READY),
            CompletePickAndPlace('Finish Pick And Place')
        ])

        system_skills_seq = py_trees.composites.Selector(
            'System Skills', memory=True)
        system_skills_seq.add_children([
            pick_seq,
            place_seq,
            complete_seq,
            IdleStatus('Idle Status')
        ])

        pick_and_place_seq = py_trees.composites.Sequence(
            'Pick And Place', memory=True)
        pick_and_place_seq.add_children([
            PickAndPlaceActive('Active?'),
            system_skills_seq
        ])

        workflow_seq = py_trees.composites.Sequence(
            'Workflow', memory=True)
        workflow_seq.add_children([
            SystemCheck("System OK?"),
            pick_and_place_seq
        ])

        fallback_seq = py_trees.composites.Sequence(
            'Fallback', memory=True)
        fallback_seq.add_children([
            PickAndPlaceActive('Active?'),
            CallService(
                'Abort Pick', '/system_skill_pick/abort', Stages.ABORT),
            CallService(
                'Abort Place', '/system_skill_place/abort', Stages.ABORT),
            CompletePickAndPlace('Abort Pick And Place')
        ])

        system_monitor_sel = py_trees.composites.Selector(
            'System Monitor', memory=True)
        system_monitor_sel.add_children([
            workflow_seq,
            fallback_seq,
        ])

        root.add_children([
            system_state,
            system_monitor_sel
        ])

        # behaviour tree
        self.tree = py_trees.trees.BehaviourTree(root)
        self.tree.setup(node=self)
        # render the tree
        py_trees.display.render_dot_tree(
            root,
            name='task_pick_and_place',
            target_directory='diagrams'
        )
        py_trees.display.render_dot_tree(
            root,
            name='task_pick_and_place_with_bb',
            target_directory='diagrams',
            with_blackboard_variables=True
        )

        self.blackboard = py_trees.blackboard.Client(name='Global')
        self.blackboard.register_key(
            key='active',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.active = False
        self.blackboard.register_key(
            key='stage',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.stage = Stages.ACTIVE

        # tick_tree timer
        self.timer = self.create_timer(1.0, self.tick_tree)

        self._start_pick_and_place_srv = self.create_service(
            Void, '/task_pick_and_place/execute', self.start_pick_and_place_callback)

        self._abort_pick_and_place_srv = self.create_service(
            Void, '/task_pick_and_place/abort', self.abort_pick_and_place_callback)

    def start_pick_and_place_callback(self, request, response):
        self.blackboard.stage = Stages.ACTIVE
        self.blackboard.active = True

        status = {
            'status': 'OK'
        }
        response.status = json.dumps(status)

        self.get_logger().info('Pick And Place started.')

        return response

    def abort_pick_and_place_callback(self, request, response):
        self.blackboard.active = False

        status = {
            'status': 'OK'
        }
        response.status = json.dumps(status)

        self.get_logger().info('Pick And Place aborted.')

        return response

    def tick_tree(self):
        self.tree.tick()


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
