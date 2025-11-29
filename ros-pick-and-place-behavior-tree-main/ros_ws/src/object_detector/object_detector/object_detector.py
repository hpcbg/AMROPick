import json

import rclpy
from rclpy.node import Node

from custom_interfaces.srv import ManipulateState

from std_msgs.msg import String


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.detected_object = {
            'found': False,
            'pose': {
                'x': 0,
                'y': 0,
                'z': 0,
                'rx': 0,
                'ry': 0,
                'rz': 0
            }
        }

        self.detected_object_publisher = self.create_publisher(
            String, '/object_detector/detected_object_json', 1)

        self._manipulate_state_srv = self.create_service(
            ManipulateState, '/object_detector/manipulate_state', self.manipulate_state_callback)

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

        self.get_logger().info('Object Detector state manipulated.')

        return response

    def main_loop(self):
        msg = String()
        msg.data = json.dumps(self.detected_object)
        self.detected_object_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    object_detector = ObjectDetector()

    rclpy.spin(object_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
