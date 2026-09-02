import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint


# ==============================================================
# UR10 MULTI-MOTION CONTROL NODE
#
# Runs a fixed sequence of motions on a UR10 arm via MoveIt 2:
#
#   1. Joint-space move to "home"
#   2. Joint-space move to "test_configuration"
#   3. A 15cm x 15cm square traced with a straight-line Cartesian
#      path, starting from wherever the arm ends up after step 2
#   4. Joint-space move to "up"
#
# The node shuts itself down automatically once the sequence
# finishes (or a step fails) - no Ctrl+C required.
# ==============================================================

class UR10MultiMotionControl(Node):

    def __init__(self):
        super().__init__('ur10_multi_motion_control')

        # --------------------------------------------------
        # MoveIt planning group / frame configuration
        # --------------------------------------------------
        self.group_name = 'ur_manipulator'
        self.end_effector_link = 'tool0'
        self.reference_frame = 'base_link'

        # Joint order used everywhere joint positions are sent
        # (must match the order the controller/driver expects).
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # --------------------------------------------------
        # Named joint-space target configurations (radians)
        # --------------------------------------------------
        self.POSITIONS = {
            'home': [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            'test_configuration': [0.0, -1.2, 1.2, -1.57, -1.57, 0.0],
            'up': [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # Fully extended vertical pose
        }

        # --------------------------------------------------
        # Action clients (MoveGroup: plan + execute joint goals,
        # ExecuteTrajectory: run an already-computed trajectory)
        # and service clients (FK: joint state -> Cartesian pose,
        # Cartesian path: compute a straight-line multi-waypoint path)
        # --------------------------------------------------
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')

        # --------------------------------------------------
        # Live joint state, kept up to date via subscription.
        # Used as the "current pose" reference when computing FK
        # and as the Cartesian path's starting robot state.
        # --------------------------------------------------
        self.current_joint_state = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # --------------------------------------------------
        # Motion tuning parameters
        # --------------------------------------------------
        self.eef_step = 0.005          # Distance (m) between interpolated Cartesian points - 5mm for higher precision
        self.jump_threshold = 0.0      # Disable jump threshold check (no protection against IK "flips" near singularities)
        self.minimum_fraction = 0.90   # Refuse to execute a Cartesian path that's less than 90% achievable

        # --------------------------------------------------
        # The sequence to run, and where we are in it.
        # 'cartesian_path' is a special step handled separately
        # from the named joint-space targets in self.POSITIONS.
        # --------------------------------------------------
        self.sequence_step = 0
        self.motion_sequence = ['home', 'test_configuration', 'cartesian_path', 'up']

        # --------------------------------------------------
        # Shutdown-control flags.
        #
        # We never call rclpy.shutdown() directly from inside a
        # callback - doing so while spin() is still running can
        # leave spin() in a bad state instead of returning cleanly.
        # Instead, callbacks call self._finish(success) to flip
        # these flags, and main() polls them in its own spin loop,
        # handling actual teardown once spin_once() has returned.
        # --------------------------------------------------
        self.task_finished = False
        self.task_succeeded = False

        # --------------------------------------------------
        # Block here (in __init__, before we start spinning) until
        # every MoveIt interface we depend on is actually available.
        # --------------------------------------------------
        self.get_logger().info('Waiting for MoveIt 2 interfaces...')
        self.move_group_client.wait_for_server()
        self.cartesian_path_client.wait_for_service()
        self.execute_trajectory_client.wait_for_server()
        self.fk_client.wait_for_service()

        self.get_logger().info('MoveIt 2 interfaces ready.')

    # ======================================================
    # JOINT STATE SUBSCRIPTION
    # ======================================================

    def joint_state_callback(self, msg):
        """
        Cache the latest /joint_states message, reordered to match
        self.joint_names. Incoming JointState messages don't
        guarantee any particular ordering of joints, so we look each
        one up by name rather than assuming index alignment.
        """
        if all(name in msg.name for name in self.joint_names):
            ordered_msg = JointState()
            ordered_msg.header = msg.header
            ordered_msg.name = self.joint_names
            ordered_msg.position = [msg.position[msg.name.index(n)] for n in self.joint_names]
            self.current_joint_state = ordered_msg

    # ======================================================
    # SEQUENCE DRIVER
    #
    # Each step calls into this once it completes, advancing
    # self.sequence_step and either finishing the whole sequence
    # or kicking off the next step.
    # ======================================================

    def execute_next_step(self):

        # No more steps left -> the whole sequence is done.
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
            # A named joint-space target ('home', 'test_configuration', 'up')
            self.move_to_joint_target(self.POSITIONS[current_target])
        elif current_target == 'cartesian_path':
            # The special square-tracing Cartesian step
            self.get_current_ee_pose_and_plan()

    # ======================================================
    # JOINT-SPACE MOTION (home / test_configuration / up)
    # ======================================================

    def move_to_joint_target(self, joint_positions):
        """
        Send a MoveGroup goal that plans and executes a free-space
        motion to the given joint positions.
        """
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = 0.4
        goal.request.max_acceleration_scaling_factor = 0.4

        # Build one joint constraint per joint - together these
        # define the target configuration MoveIt should plan to.
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
        goal.planning_options.plan_only = False  # plan AND execute in one goal

        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(self.joint_goal_response)

    def joint_goal_response(self, future):
        """Called once MoveIt accepts/rejects the joint goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Joint goal rejected.')
            self._finish(success=False)
            return

        # Goal accepted - now wait for it to actually finish executing.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.joint_result_callback)

    def joint_result_callback(self, future):
        """Called once the joint-space motion has finished executing."""
        result = future.result().result
        if result.error_code.val != 1:  # 1 == SUCCESS in MoveItErrorCodes
            self.get_logger().error(f'Joint execution failed code: {result.error_code.val}')
            self._finish(success=False)
            return

        self.get_logger().info(f'Reached target configuration: {self.motion_sequence[self.sequence_step]}')
        self.sequence_step += 1

        # Brief pause before starting the next step (lets the arm
        # settle, and gives /joint_states a moment to catch up).
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        """One-shot timer callback: cancel itself, then continue the sequence."""
        self.timer.cancel()
        self.execute_next_step()

    # ======================================================
    # CARTESIAN SQUARE STEP
    #
    # Traces a 15cm x 15cm square in the XY plane, starting from
    # wherever the end effector currently is, as ONE continuous
    # straight-line Cartesian path (not planned corner-to-corner).
    # ======================================================

    def get_current_ee_pose_and_plan(self):
        """
        Step 1 of the Cartesian square: ask MoveIt for the end
        effector's current Cartesian pose via forward kinematics,
        so the square can be built relative to wherever the arm
        actually is right now.
        """
        req = GetPositionFK.Request()
        req.header.frame_id = self.reference_frame
        req.fk_link_names = [self.end_effector_link]
        if self.current_joint_state is not None:
            req.robot_state.joint_state = self.current_joint_state

        future = self.fk_client.call_async(req)
        future.add_done_callback(self.fk_callback)

    def fk_callback(self, future):
        """
        Step 2 of the Cartesian square: build the four corners of a
        15cm x 15cm square (plus the starting point, so the path
        closes back on itself) and request a single straight-line
        Cartesian path through all of them.
        """
        res = future.result()
        current_pose = res.pose_stamped[0].pose

        x0, y0, z0 = current_pose.position.x, current_pose.position.y, current_pose.position.z

        # 15 cm x 15 cm Reachable Cartesian Square Pattern:
        #   wp0: starting corner (current pose)
        #   wp1: +X corner
        #   wp2: +X, +Y corner (diagonal from start)
        #   wp3: +Y corner
        #   wp4: back to the starting corner, closing the square
        # Orientation is held fixed at the current orientation throughout.
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

        # Anchor the path's starting state to the robot's actual
        # current joint state, so the computed trajectory starts
        # exactly where the arm currently is.
        if self.current_joint_state is not None:
            req.start_state.joint_state = self.current_joint_state

        future_path = self.cartesian_path_client.call_async(req)
        future_path.add_done_callback(self.cartesian_path_callback)

    def cartesian_path_callback(self, future):
        """
        Step 3 of the Cartesian square: inspect how much of the
        requested path MoveIt was actually able to compute
        (`fraction`), and either execute it or abort if too much
        of the path was unreachable.
        """
        res = future.result()
        fraction = res.fraction
        self.get_logger().info(f'Cartesian Path computed fraction: {fraction * 100.0:.1f}%')

        if fraction < self.minimum_fraction:
            self.get_logger().error(f'Cartesian computation incomplete: {fraction * 100.0:.1f}%')
            self._finish(success=False)
            return

        # Enough of the path was achievable - execute the resulting
        # trajectory as a single continuous motion.
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        future_exec = self.execute_trajectory_client.send_goal_async(goal)
        future_exec.add_done_callback(self.execute_goal_response)

    def execute_goal_response(self, future):
        """Called once MoveIt accepts/rejects the trajectory execution goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Cartesian trajectory execution goal rejected.')
            self._finish(success=False)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.cartesian_result_callback)

    def cartesian_result_callback(self, future):
        """
        Step 4 of the Cartesian square: called once the trajectory
        has finished executing. On success, this is also the last
        step in the overall sequence before "up", so it advances
        sequence_step and continues on to the next step.
        """
        result = future.result().result
        if result.error_code.val != 1:  # 1 == SUCCESS in MoveItErrorCodes
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
        """
        Mark the task as done. This only flips a flag - it does NOT
        call rclpy.shutdown() from inside a callback. main() polls
        this flag while spinning and handles teardown from the
        top-level thread once spin() has actually returned.
        """
        self.task_succeeded = success
        self.task_finished = True


# ==============================================================
# MAIN
# ==============================================================

def main(args=None):

    rclpy.init(args=args)

    node = UR10MultiMotionControl()

    # Kick off the first step of the sequence. Everything after
    # this happens asynchronously, driven by callbacks.
    node.execute_next_step()

    # --------------------------------------------------
    # Spin until the sequence flags itself done (success or
    # failure), rather than blocking forever on rclpy.spin().
    # This is what lets the node shut itself down automatically
    # instead of requiring Ctrl+C.
    # --------------------------------------------------
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

    # Force the process to exit even if some background thread from
    # the action/service clients is still alive.
    sys.exit(0 if node.task_succeeded else 1)


if __name__ == '__main__':
    main()