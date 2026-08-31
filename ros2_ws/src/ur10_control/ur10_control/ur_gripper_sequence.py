import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# ==============================================================
# UR10 + GRIPPER COMBINED SEQUENCE NODE
#
#   1. Joint-space move to "home"
#   2. Joint-space move to "test_configuration"  -> gripper CLOSES
#   3. 15cm x 15cm Cartesian square (unchanged)
#   4. Joint-space move to "up"                  -> gripper OPENS
#
# Gripper motion is inserted as a blocking step immediately after
# the arm reaches the relevant joint-space target, before the
# sequence advances further.
# ==============================================================

GRIPPER_JOINTS = [
    "gripper_finger_1_joint_1",
    "gripper_finger_2_joint_1",
    "gripper_finger_middle_joint_1",
]

GRIPPER_OPEN_POSITIONS = [0.05, 0.05, 0.05]
CLOSE_SIDE = 0.7
CLOSE_MIDDLE = 0.7
GRIPPER_CLOSE_POSITIONS = [CLOSE_SIDE, CLOSE_SIDE, CLOSE_MIDDLE]

GRIPPER_NUM_WAYPOINTS = 8
GRIPPER_TOTAL_DURATION_SEC = 1.6


def ease_in_out(t: float) -> float:
    """Smoothstep-style easing, t in [0, 1] -> eased fraction in [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


def build_eased_points(start, end, num_waypoints, total_duration_sec):
    """Eased multi-waypoint trajectory with finite-difference velocities,
    same profile as the standalone gripper_test.py script."""
    times = [
        (i / num_waypoints) * total_duration_sec for i in range(1, num_waypoints + 1)
    ]
    fractions = [ease_in_out(i / num_waypoints) for i in range(1, num_waypoints + 1)]

    positions_by_point = []
    for frac in fractions:
        positions_by_point.append([s + (e - s) * frac for s, e in zip(start, end)])

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


class UR10GripperSequence(Node):

    # Maps an arm target name -> gripper action to run once that
    # target is reached ('close' or 'open'). Targets not listed here
    # get no gripper action.
    GRIPPER_ACTIONS = {
        'test_configuration': 'close',
        'up': 'open',
    }

    def __init__(self):
        super().__init__('ur10_gripper_sequence')

        self.group_name = 'ur_manipulator'
        self.end_effector_link = 'tool0'
        self.reference_frame = 'base_link'

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.POSITIONS = {
            'home': [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            'test_configuration': [0.0, -1.2, 1.2, -1.57, -1.57, 0.0],
            'up': [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        }

        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory'
        )

        self.current_joint_state = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.eef_step = 0.005
        self.jump_threshold = 0.0
        self.minimum_fraction = 0.90

        self.sequence_step = 0
        self.motion_sequence = ['home', 'test_configuration', 'cartesian_path', 'up']

        self.task_finished = False
        self.task_succeeded = False

        self.get_logger().info('Waiting for MoveIt 2 and gripper interfaces...')
        self.move_group_client.wait_for_server()
        self.cartesian_path_client.wait_for_service()
        self.execute_trajectory_client.wait_for_server()
        self.fk_client.wait_for_service()
        self.gripper_client.wait_for_server()
        self.get_logger().info('All interfaces ready.')

    # ======================================================
    # JOINT STATE SUBSCRIPTION
    # ======================================================

    def joint_state_callback(self, msg):
        if all(name in msg.name for name in self.joint_names):
            ordered_msg = JointState()
            ordered_msg.header = msg.header
            ordered_msg.name = self.joint_names
            ordered_msg.position = [msg.position[msg.name.index(n)] for n in self.joint_names]
            self.current_joint_state = ordered_msg

    # ======================================================
    # SEQUENCE DRIVER
    # ======================================================

    def execute_next_step(self):
        if self.sequence_step >= len(self.motion_sequence):
            self.get_logger().info('============================================')
            self.get_logger().info('ALL SEQUENCE MOTIONS COMPLETED SUCCESSFULLY')
            self.get_logger().info('============================================')
            self._finish(success=True)
            return

        current_target = self.motion_sequence[self.sequence_step]
        self.get_logger().info(
            f'---> Step {self.sequence_step + 1}/{len(self.motion_sequence)}: '
            f'Moving to "{current_target}"'
        )

        if current_target in self.POSITIONS:
            self.move_to_joint_target(self.POSITIONS[current_target])
        elif current_target == 'cartesian_path':
            self.get_current_ee_pose_and_plan()

    # ======================================================
    # JOINT-SPACE MOTION (home / test_configuration / up)
    # ======================================================

    def move_to_joint_target(self, joint_positions):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = 0.4
        goal.request.max_acceleration_scaling_factor = 0.4

        constraints = Constraints()
        for name, position in zip(self.joint_names, joint_positions):
            c = JointConstraint()
            c.joint_name = name
            c.position = position
            c.tolerance_above = 0.01
            c.tolerance_below = 0.01
            c.weight = 1.0
            constraints.joint_constraints.append(c)

        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = False

        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(self.joint_goal_response)

    def joint_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Joint goal rejected.')
            self._finish(success=False)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.joint_result_callback)

    def joint_result_callback(self, future):
        result = future.result().result
        if result.error_code.val != 1:  # 1 == SUCCESS in MoveItErrorCodes
            self.get_logger().error(f'Joint execution failed code: {result.error_code.val}')
            self._finish(success=False)
            return

        target_name = self.motion_sequence[self.sequence_step]
        self.get_logger().info(f'Reached target configuration: {target_name}')

        gripper_action = self.GRIPPER_ACTIONS.get(target_name)
        if gripper_action:
            # Blocking gripper step: don't advance sequence_step or
            # start the settle timer until the gripper motion finishes.
            self.send_gripper_goal(gripper_action)
        else:
            self.sequence_step += 1
            self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.timer.cancel()
        self.execute_next_step()

    # ======================================================
    # GRIPPER MOTION (open / close)
    # ======================================================

    def send_gripper_goal(self, action):
        start = GRIPPER_OPEN_POSITIONS if action == 'close' else GRIPPER_CLOSE_POSITIONS
        end = GRIPPER_CLOSE_POSITIONS if action == 'close' else GRIPPER_OPEN_POSITIONS

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINTS

        points = []
        for positions, velocities, t in build_eased_points(
            start, end, GRIPPER_NUM_WAYPOINTS, GRIPPER_TOTAL_DURATION_SEC
        ):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities
            sec = int(t)
            nanosec = int((t - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            points.append(point)

        goal.trajectory.points = points
        self.get_logger().info(f'Sending gripper "{action}" goal...')
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self.gripper_goal_response)

    def gripper_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected.')
            self._finish(success=False)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.gripper_result_callback)

    def gripper_result_callback(self, future):
        result = future.result().result
        if result.error_code != 0:  # 0 == SUCCESSFUL in FollowJointTrajectoryResult
            self.get_logger().error(f'Gripper motion failed, error code: {result.error_code}')
            self._finish(success=False)
            return

        self.get_logger().info('Gripper motion complete.')
        self.sequence_step += 1
        self.timer = self.create_timer(1.0, self.timer_callback)

    # ======================================================
    # CARTESIAN SQUARE STEP (unchanged)
    # ======================================================

    def get_current_ee_pose_and_plan(self):
        req = GetPositionFK.Request()
        req.header.frame_id = self.reference_frame
        req.fk_link_names = [self.end_effector_link]
        if self.current_joint_state is not None:
            req.robot_state.joint_state = self.current_joint_state

        future = self.fk_client.call_async(req)
        future.add_done_callback(self.fk_callback)

    def fk_callback(self, future):
        res = future.result()
        current_pose = res.pose_stamped[0].pose

        x0, y0, z0 = current_pose.position.x, current_pose.position.y, current_pose.position.z

        wp0 = current_pose
        wp1 = Pose(orientation=current_pose.orientation, position=Point(x=x0 + 0.15, y=y0, z=z0))
        wp2 = Pose(orientation=current_pose.orientation, position=Point(x=x0 + 0.15, y=y0 + 0.15, z=z0))
        wp3 = Pose(orientation=current_pose.orientation, position=Point(x=x0, y=y0 + 0.15, z=z0))
        wp4 = Pose(orientation=current_pose.orientation, position=Point(x=x0, y=y0, z=z0))

        req = GetCartesianPath.Request()
        req.header.frame_id = self.reference_frame
        req.group_name = self.group_name
        req.link_name = self.end_effector_link
        req.waypoints = [wp0, wp1, wp2, wp3, wp4]
        req.max_step = self.eef_step
        req.jump_threshold = self.jump_threshold
        req.avoid_collisions = True

        if self.current_joint_state is not None:
            req.start_state.joint_state = self.current_joint_state

        future_path = self.cartesian_path_client.call_async(req)
        future_path.add_done_callback(self.cartesian_path_callback)

    def cartesian_path_callback(self, future):
        res = future.result()
        fraction = res.fraction
        self.get_logger().info(f'Cartesian Path computed fraction: {fraction * 100.0:.1f}%')

        if fraction < self.minimum_fraction:
            self.get_logger().error(f'Cartesian computation incomplete: {fraction * 100.0:.1f}%')
            self._finish(success=False)
            return

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        future_exec = self.execute_trajectory_client.send_goal_async(goal)
        future_exec.add_done_callback(self.execute_goal_response)

    def execute_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Cartesian trajectory execution goal rejected.')
            self._finish(success=False)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.cartesian_result_callback)

    def cartesian_result_callback(self, future):
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().error(f'Cartesian trajectory execution failed code: {result.error_code.val}')
            self._finish(success=False)
            return

        self.get_logger().info('Completed Cartesian Path Step.')
        self.sequence_step += 1
        self.execute_next_step()

    # ======================================================
    # FINISH HELPER
    # ======================================================

    def _finish(self, success):
        self.task_succeeded = success
        self.task_finished = True


def main(args=None):
    rclpy.init(args=args)
    node = UR10GripperSequence()
    node.execute_next_step()

    try:
        while rclpy.ok() and not node.task_finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down node...')
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(0 if node.task_succeeded else 1)


if __name__ == '__main__':
    main()