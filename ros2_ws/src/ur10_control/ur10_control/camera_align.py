import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped


class UR10CameraAlign(Node):

    def __init__(self):
        super().__init__('ur10_camera_align')

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'
        )

        self.move_group_client.wait_for_server()

    def create_goal(self, target_pose: PoseStamped, link_name: str):

        goal = MoveGroup.Goal()

        goal.request.group_name = 'ur_manipulator'
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.5

        constraints = Constraints()

        # Position constraint -- small box tolerance around target xyz
        position_constraint = PositionConstraint()
        position_constraint.header = target_pose.header
        position_constraint.link_name = link_name

        bounding_volume = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.02, 0.02, 0.02]
        bounding_volume.primitives.append(primitive)
        bounding_volume.primitive_poses.append(target_pose.pose)

        position_constraint.constraint_region = bounding_volume
        position_constraint.weight = 1.0

        # Orientation constraint -- camera optical Z pointing straight down
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = target_pose.header
        orientation_constraint.link_name = link_name
        orientation_constraint.orientation = target_pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)

        goal.request.goal_constraints.append(constraints)

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True

        return goal

    def move_to_pose(self, target_pose, link_name):

        goal = self.create_goal(target_pose, link_name)

        send_goal_future = self.move_group_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            rclpy.shutdown()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):

        result = future.result().result
        error_code = result.error_code.val

        if error_code == 1:
            self.get_logger().info('Camera aligned with target successfully')
        else:
            self.get_logger().error(
                f'Planning/execution failed, error code: {error_code}'
            )

        rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = UR10CameraAlign()

    target_pose = PoseStamped()
    target_pose.header.frame_id = 'base_link'

    # Directly above the test cube (0.5, 0, 0.1), 0.3m standoff
    target_pose.pose.position.x = 0.5
    target_pose.pose.position.y = 0.0
    target_pose.pose.position.z = 0.4

    # 180 deg rotation about X: optical Z axis -> straight down
    target_pose.pose.orientation.x = -0.7071
    target_pose.pose.orientation.y = 0.0
    target_pose.pose.orientation.z = 0.0
    target_pose.pose.orientation.w = 0.7071

    node.move_to_pose(target_pose, 'tool0')

    rclpy.spin(node)


if __name__ == '__main__':
    main()