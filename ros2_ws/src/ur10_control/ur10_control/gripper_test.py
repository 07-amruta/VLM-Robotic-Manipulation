#!/usr/bin/env python3
"""
Open/close test for the 3F gripper's gripper_controller
(joint_trajectory_controller). No MoveIt involved -- direct action
client, same pattern as your arm control scripts.

Sends a multi-waypoint eased trajectory (smooth accel/decel) instead of a
single jump, so the motion reads closer to a real gripper's motor ramp.

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

JOINTS = [
    "gripper_finger_1_joint_1",
    "gripper_finger_2_joint_1",
    "gripper_finger_middle_joint_1",
]

OPEN_POSITIONS = [0.05, 0.05, 0.05]
# Side fingers and middle finger can be tuned independently — start here and
# adjust based on what you see, see tuning note below.
CLOSE_SIDE = 0.3
CLOSE_MIDDLE = 0.3
CLOSE_POSITIONS = [CLOSE_SIDE, CLOSE_SIDE, CLOSE_MIDDLE]

# Number of intermediate waypoints (not counting the start) and total duration.
NUM_WAYPOINTS = 8
TOTAL_DURATION_SEC = 1.6


def ease_in_out(t: float) -> float:
    """Smoothstep-style easing, t in [0, 1] -> eased fraction in [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


def build_eased_points(start, end, num_waypoints, total_duration_sec):
    """Build a list of (positions, time_from_start_sec) using an eased profile,
    plus per-point velocities estimated by finite difference so the controller
    spline doesn't overshoot or jerk at each waypoint."""
    times = [
        (i / num_waypoints) * total_duration_sec for i in range(1, num_waypoints + 1)
    ]
    fractions = [ease_in_out(i / num_waypoints) for i in range(1, num_waypoints + 1)]

    positions_by_point = []
    for frac in fractions:
        positions_by_point.append([s + (e - s) * frac for s, e in zip(start, end)])

    # Estimate velocities at each waypoint via central differences, zero at the
    # final point so the motion settles instead of ending abruptly.
    velocities_by_point = []
    for i in range(len(positions_by_point)):
        if i == len(positions_by_point) - 1:
            velocities_by_point.append([0.0] * len(start))
            continue
        prev_pos = positions_by_point[i - 1] if i > 0 else start
        next_pos = positions_by_point[i + 1]
        prev_t = times[i - 1] if i > 0 else 0.0
        next_t = times[i + 1]
        dt = max(next_t - prev_t, 1e-6)
        velocities_by_point.append(
            [(n - p) / dt for n, p in zip(next_pos, prev_pos)]
        )

    return list(zip(positions_by_point, velocities_by_point, times))


class GripperTestClient(Node):
    def __init__(self):
        super().__init__("gripper_test_client")
        self._client = ActionClient(
            self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )

    def send_eased_goal(self, start, end):
        self._client.wait_for_server(timeout_sec=5.0)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS

        points = []
        for positions, velocities, t in build_eased_points(
            start, end, NUM_WAYPOINTS, TOTAL_DURATION_SEC
        ):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities
            sec = int(t)
            nanosec = int((t - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            points.append(point)

        goal.trajectory.points = points
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
    if mode == "open":
        node.send_eased_goal(CLOSE_POSITIONS, OPEN_POSITIONS)
    else:
        node.send_eased_goal(OPEN_POSITIONS, CLOSE_POSITIONS)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()