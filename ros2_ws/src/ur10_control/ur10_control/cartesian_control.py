import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint
)

from geometry_msgs.msg import Pose


class UR10CartesianControl(Node):

    def __init__(self):
        super().__init__('ur10_cartesian_control')

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'
        )

        self.move_group_client.wait_for_server()

    def create_goal(self, pose):

        goal = MoveGroup.Goal()

        # Planning group
        goal.request.group_name = 'ur_manipulator'

        # Motion scaling
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 1.0

        # --------------------------------------------------
        # Cartesian constraints
        # --------------------------------------------------

        constraints = Constraints()

        # Position constraint
        position_constraint = PositionConstraint()

        position_constraint.header.frame_id = 'base_link'
        position_constraint.link_name = 'tool0'

        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        position_constraint.constraint_region.primitive_poses.append(
            pose
        )

        position_constraint.constraint_region.primitives.append(
            self.create_bounding_box()
        )

        position_constraint.weight = 1.0

        constraints.position_constraints.append(
            position_constraint
        )

        # Orientation constraint
        orientation_constraint = OrientationConstraint()

        orientation_constraint.header.frame_id = 'base_link'
        orientation_constraint.link_name = 'tool0'

        orientation_constraint.orientation = pose.orientation

        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01

        orientation_constraint.weight = 1.0

        constraints.orientation_constraints.append(
            orientation_constraint
        )

        goal.request.goal_constraints.append(
            constraints
        )

        # Planning options
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True

        return goal

    def create_bounding_box(self):

        from shape_msgs.msg import SolidPrimitive

        box = SolidPrimitive()

        box.type = SolidPrimitive.BOX

        # Small region around target position
        box.dimensions = [
            0.01,
            0.01,
            0.01
        ]

        return box

    def move_to_pose(self, pose):

        goal = self.create_goal(pose)

        send_goal_future = (
            self.move_group_client.send_goal_async(goal)
        )

        send_goal_future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            rclpy.shutdown()
            return

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.result_callback
        )

    def result_callback(self, future):

        result = future.result().result

        if result.error_code.val == 1:
            print('Cartesian motion completed successfully.')

        else:
            print(
                'Cartesian motion failed. '
                f'MoveIt error code: {result.error_code.val}'
            )

        rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = UR10CartesianControl()

    # --------------------------------------------------
    # Current pose obtained from TF
    # --------------------------------------------------

    target_pose = Pose()

    # Move +5 cm in X
    target_pose.position.x = -0.05
    target_pose.position.y = 0.256
    target_pose.position.z = 1.427

    # Current tool0 orientation
    target_pose.orientation.x = -0.707
    target_pose.orientation.y = 0.0
    target_pose.orientation.z = 0.0
    target_pose.orientation.w = 0.707

    node.move_to_pose(target_pose)

    rclpy.spin(node)


if __name__ == '__main__':
    main()