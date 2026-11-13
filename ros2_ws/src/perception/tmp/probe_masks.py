import rclpy
from rclpy.node import Node
from perception_msgs.msg import InstanceMaskArray


class Probe(Node):
    def __init__(self):
        super().__init__('probe_masks')
        self.create_subscription(InstanceMaskArray, '/instance_masks/yolo', self.cb, 5)

    def cb(self, m):
        self.get_logger().info(
            f'n={len(m.instances)} {[(i.class_name, round(i.score, 2), i.track_id) for i in m.instances]}')


rclpy.init()
rclpy.spin(Probe())