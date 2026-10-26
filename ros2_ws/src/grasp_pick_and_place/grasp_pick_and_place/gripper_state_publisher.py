import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from grasp_pick_and_place.config import GRIPPER, GRIPPER_JOINTS

RATE = 10.0  # Hz


class GripperStatePublisher(Node):
    """Publishes the finger joints of a gripper without joint-state feedback, at the open pose."""

    def __init__(self):
        super().__init__('gripper_state_publisher')
        self.msg = JointState(
            name=list(GRIPPER_JOINTS), position=[float(p) for p in GRIPPER['gripper_open']])
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_timer(1.0 / RATE, self.publish)

    def publish(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


def main():
    rclpy.init()
    node = GripperStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()