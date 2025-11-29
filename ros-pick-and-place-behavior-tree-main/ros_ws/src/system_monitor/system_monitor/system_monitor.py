import json

import rclpy
from rclpy.node import Node

from custom_interfaces.srv import ManipulateState

from std_msgs.msg import String


class SystemMonitor(Node):

    def __init__(self):
        super().__init__('system_monitor')

        self.status = 'ok'

        self.status_publisher = self.create_publisher(
            String, '/system_monitor/status_json', 1)

        self._manipulate_state_srv = self.create_service(
            ManipulateState, '/system_monitor/manipulate_state', self.manipulate_state_callback)

        self.timer = self.create_timer(1.0, self.main_loop)

    def manipulate_state_callback(self, request, response):
        manipulated_state = json.loads(request.manipulated_state_json)

        for k in manipulated_state:
            try:
                self.__dict__[k] = manipulated_state[k]
            except:
                print(k)

        status = {
            'status': 'OK'
        }
        response.status = json.dumps(status)

        self.get_logger().info('System Monitor state manipulated.')

        return response

    def main_loop(self):
        msg = String()
        msg.data = json.dumps(self.status)
        self.status_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    system_monitor = SystemMonitor()

    rclpy.spin(system_monitor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    system_monitor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
