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

GRIPPER_JOINTS = [
    "gripper_finger_1_joint_1",
    "gripper_finger_2_joint_1",
    "gripper_finger_middle_joint_1",
]

# Restore the official Robotiq 3F geometry and apply the opposite sign to the
# middle finger command so the middle finger closes/open in the physical opposite
# direction while the side fingers move together.
OPEN_POSITIONS = [0.05, 0.05, -0.05]
CLOSE_POSITIONS = [1.0, 1.0, -1.0]


class GripperTestClient(Node):
    def __init__(self):
        super().__init__("gripper_test_client")
        self._client = ActionClient(
            self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )

    def send_goal(self, positions, duration_sec=2.0):
        self._client.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration_sec))
        goal.trajectory.points = [point]
        future = self._client.send_goal_async(goal)
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
    node.send_goal(OPEN_POSITIONS if mode == "open" else CLOSE_POSITIONS)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()