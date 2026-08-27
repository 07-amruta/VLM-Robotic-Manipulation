#!/usr/bin/env python3
"""
Minimal open/close test for the 3F gripper's gripper_controller
(joint_trajectory_controller). No MoveIt involved -- direct action
client, same pattern as your arm control scripts.

Usage:
    ros2 run ur10_control gripper_test.py open
    ros2 run ur10_control gripper_test.py close
"""
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from ur10_control.gripper_middle_joint_controller import Robotiq3FGripperController

SIDE_JOINTS = [
    "gripper_finger_1_joint_1",
    "gripper_finger_2_joint_1",
]
MIDDLE_JOINTS = [
    "gripper_finger_middle_joint_1",
]

# Preserve the Robotiq 3F middle-finger mounting without any mirrored frame or
# flipped axis. The side fingers remain as one coordinated pair, while the middle
# finger keeps its own independent command trajectory and physical orientation.
controller = Robotiq3FGripperController()
OPEN_SIDE = controller.side_fingers.command(0.05)
OPEN_MIDDLE = controller.middle_joint.command(0.05)
CLOSE_SIDE = controller.side_fingers.command(1.0)
CLOSE_MIDDLE = controller.middle_joint.command(1.0)


class GripperTestClient(Node):
    def __init__(self):
        super().__init__("gripper_test_client")
        self._client = ActionClient(
            self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )

    def send_goal(self, controller_name, joint_names, positions, duration_sec=2.0):
        client = ActionClient(self, FollowJointTrajectory, controller_name)
        client.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(sec=int(duration_sec))
        goal.trajectory.points = [point]
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Gripper motion complete")


def main():
    rclpy.init()
    mode = sys.argv[1] if len(sys.argv) > 1 else "open"
    node = GripperTestClient()
    side_positions = OPEN_SIDE if mode == "open" else CLOSE_SIDE
    middle_positions = OPEN_MIDDLE if mode == "open" else CLOSE_MIDDLE
    node.send_goal(
        "/gripper_controller/follow_joint_trajectory",
        SIDE_JOINTS,
        side_positions,
    )
    node.send_goal(
        "/gripper_middle_joint_controller/follow_joint_trajectory",
        MIDDLE_JOINTS,
        [middle_positions],
    )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()