import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint
)
from geometry_msgs.msg import Pose


class UR10JointCartesianControl(Node):

    def __init__(self):

        super().__init__('ur10_joint_cartesian_control')

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'
        )

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.get_logger().info(
            'Waiting for MoveIt 2...'
        )

        self.move_group_client.wait_for_server()

        self.get_logger().info(
            'MoveIt 2 MoveGroup action connected.'
        )

    # ======================================================
    # JOINT GOAL
    # ======================================================

    def create_joint_goal(self, joint_positions):

        goal = MoveGroup.Goal()

        goal.request.group_name = 'ur_manipulator'

        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 1.0

        constraints = Constraints()

        for name, position in zip(
            self.joint_names,
            joint_positions
        ):

            constraint = JointConstraint()

            constraint.joint_name = name
            constraint.position = position

            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0

            constraints.joint_constraints.append(
                constraint
            )

        goal.request.goal_constraints.append(
            constraints
        )

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True

        return goal

    # ======================================================
    # CARTESIAN POSE GOAL
    # ======================================================

    def create_cartesian_goal(self, pose):

        goal = MoveGroup.Goal()

        goal.request.group_name = 'ur_manipulator'

        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.5

        constraints = Constraints()

        # --------------------------------------------------
        # Position
        # --------------------------------------------------

        position_constraint = PositionConstraint()

        position_constraint.header.frame_id = 'base_link'
        position_constraint.link_name = 'tool0'

        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        # Use a very small tolerance around the target
        position_constraint.weight = 1.0

        # --------------------------------------------------
        # Orientation
        # --------------------------------------------------

        orientation_constraint = OrientationConstraint()

        orientation_constraint.header.frame_id = 'base_link'
        orientation_constraint.link_name = 'tool0'

        orientation_constraint.orientation = pose.orientation

        orientation_constraint.absolute_x_axis_tolerance = 0.02
        orientation_constraint.absolute_y_axis_tolerance = 0.02
        orientation_constraint.absolute_z_axis_tolerance = 0.02

        orientation_constraint.weight = 1.0

        constraints.position_constraints.append(
            position_constraint
        )

        constraints.orientation_constraints.append(
            orientation_constraint
        )

        goal.request.goal_constraints.append(
            constraints
        )

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True

        return goal

    # ======================================================
    # SEND JOINT MOTION
    # ======================================================

    def move_to_joint_position(self, joint_positions):

        self.get_logger().info(
            'Moving to START joint position...'
        )

        goal = self.create_joint_goal(joint_positions)

        future = self.move_group_client.send_goal_async(goal)

        future.add_done_callback(
            self.joint_goal_response
        )

    def joint_goal_response(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.get_logger().error(
                'START joint goal rejected.'
            )

            rclpy.shutdown()
            return

        self.get_logger().info(
            'START joint goal accepted.'
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.joint_result_callback
        )

    def joint_result_callback(self, future):

        result = future.result().result

        if result.error_code.val != 1:

            self.get_logger().error(
                f'Joint motion failed: '
                f'{result.error_code.val}'
            )

            rclpy.shutdown()
            return

        self.get_logger().info(
            'START joint position reached.'
        )

        # Continue to Cartesian motion
        self.execute_cartesian_sequence()

    # ======================================================
    # CARTESIAN MOTION
    # ======================================================

    def execute_cartesian_sequence(self):

        self.get_logger().info(
            'Starting Cartesian motion...'
        )

        # --------------------------------------------------
        # Example Cartesian target
        #
        # These values are expressed in base_link.
        # Change them according to your workspace.
        # --------------------------------------------------

        self.cartesian_waypoints = [

            # Waypoint 1
            {
                'x': 0.10,
                'y': 0.256,
                'z': 1.427
            },

            # Waypoint 2
            {
                'x': 0.20,
                'y': 0.256,
                'z': 1.427
            },

            # Final target
            {
                'x': 0.20,
                'y': 0.356,
                'z': 1.427
            }
        ]

        self.current_waypoint = 0

        # Fixed orientation
        self.orientation = (
            -0.707,
            0.0,
            0.0,
            0.707
        )

        self.move_to_next_waypoint()

    # ======================================================
    # NEXT WAYPOINT
    # ======================================================

    def move_to_next_waypoint(self):

        if self.current_waypoint >= len(
            self.cartesian_waypoints
        ):

            self.get_logger().info(
                'Joint + Cartesian sequence completed.'
            )

            rclpy.shutdown()
            return

        waypoint = self.cartesian_waypoints[
            self.current_waypoint
        ]

        pose = Pose()

        pose.position.x = waypoint['x']
        pose.position.y = waypoint['y']
        pose.position.z = waypoint['z']

        pose.orientation.x = self.orientation[0]
        pose.orientation.y = self.orientation[1]
        pose.orientation.z = self.orientation[2]
        pose.orientation.w = self.orientation[3]

        self.get_logger().info(
            f'Moving to Cartesian waypoint '
            f'{self.current_waypoint + 1}...'
        )

        goal = self.create_cartesian_goal(pose)

        future = self.move_group_client.send_goal_async(
            goal
        )

        future.add_done_callback(
            self.cartesian_goal_response
        )

    # ======================================================
    # CARTESIAN GOAL RESPONSE
    # ======================================================

    def cartesian_goal_response(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.get_logger().error(
                'Cartesian goal rejected.'
            )

            rclpy.shutdown()
            return

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.cartesian_result_callback
        )

    # ======================================================
    # CARTESIAN RESULT
    # ======================================================

    def cartesian_result_callback(self, future):

        result = future.result().result

        if result.error_code.val != 1:

            self.get_logger().error(
                f'Cartesian motion failed: '
                f'{result.error_code.val}'
            )

            rclpy.shutdown()
            return

        self.get_logger().info(
            f'Cartesian waypoint '
            f'{self.current_waypoint + 1} reached.'
        )

        self.current_waypoint += 1

        self.move_to_next_waypoint()


# ==========================================================
# MAIN
# ==========================================================

def main(args=None):

    rclpy.init(args=args)

    node = UR10JointCartesianControl()

    # ======================================================
    # START JOINT POSITION
    # ======================================================

    node.start_position = [
        -3.38,
        -1.57,
        2.18,
        -1.57,
        -8.79,
        -1.79
    ]

    # ======================================================
    # START MOTION
    # ======================================================

    node.move_to_joint_position(
        node.start_position
    )

    rclpy.spin(node)


if __name__ == '__main__':
    main()