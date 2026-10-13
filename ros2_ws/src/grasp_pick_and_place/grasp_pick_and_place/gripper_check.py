import rclpy
from rclpy.node import Node

from grasp_pick_and_place.config import PARAMS
from grasp_pick_and_place.gripper_real import RealGripperMixin


class GripperCheck(RealGripperMixin, Node):
    def __init__(self):
        super().__init__('gripper_check')
        for name, value in PARAMS.items():
            self.declare_parameter(name, value)


def main():
    rclpy.init()
    node = GripperCheck()
    log = node.get_logger()
    opened = node.get_parameter('gripper_open').value
    try:
        log.info(f'open: {node.move_gripper(opened)}')
        input('Place the object between the fingers, then Enter ')
        log.info(f'held: {node.close_gripper()}')
        input('Enter to open ')
        log.info(f'open: {node.move_gripper(opened)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()