import math
import sys
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

from grasp_pick_and_place.arm_check import ACTION, go
from grasp_pick_and_place.config import ARM_JOINTS, PARAMS
from grasp_pick_and_place.gripper_real import RealGripperMixin

JOINT = 'elbow_joint'
DELTA = math.radians(-10.0)
HOLD = 3.0  # s, pause at the raised pose


class HandshakeCheck(RealGripperMixin, Node):
    def __init__(self):
        super().__init__('handshake_check')
        for name, value in PARAMS.items():
            self.declare_parameter(name, value)


def main():
    rclpy.init()
    node = HandshakeCheck()
    log = node.get_logger()
    opened = node.get_parameter('gripper_open').value
    state = {}
    node.create_subscription(
        JointState, '/joint_states', lambda m: state.update(zip(m.name, m.position)), 10)
    try:
        while rclpy.ok() and not all(j in state for j in ARM_JOINTS):
            rclpy.spin_once(node, timeout_sec=0.1)
        start = [state[j] for j in ARM_JOINTS]
        target = list(start)
        index = ARM_JOINTS.index(JOINT)
        target[index] += DELTA

        client = ActionClient(node, FollowJointTrajectory, ACTION)
        if not client.wait_for_server(timeout_sec=5.0):
            log.error(f'{ACTION} not available')
            return
        log.info(f'{JOINT}: {start[index]:.3f} -> {target[index]:.3f} rad')
        if '--dry' in sys.argv:
            log.info('dry run: nothing sent, gripper untouched')
            return

        log.info(f'open: {node.move_gripper(opened)}')
        input('Object between the fingers, workspace clear, e-stop in reach. Enter to close ')
        if not node.close_gripper():
            log.error('not held: arm not moved')
        else:
            log.info(f'raise: {go(node, client, target)}')
            time.sleep(HOLD)
            log.info(f'lower: {go(node, client, start)}')
        input('Support the object, then Enter to open ')
        log.info(f'open: {node.move_gripper(opened)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()