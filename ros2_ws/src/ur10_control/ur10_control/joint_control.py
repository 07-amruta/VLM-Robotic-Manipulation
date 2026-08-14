import rclpy
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint

from rclpy.action import ActionClient


class UR10JointControl(Node):

    def __init__(self):
        super().__init__('ur10_joint_control')

        self.action_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'
        )

        self.get_logger().info('UR10 Joint Control Node started')

    def move_to_joint_position(self, joint_positions):

        self.get_logger().info('Waiting for MoveIt...')

        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'MoveIt /move_action server not available'
            )
            return

        goal = MoveGroup.Goal()

        goal.request.group_name = 'ur_manipulator'

        constraints = Constraints()

        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        for name, position in zip(joint_names, joint_positions):

            constraint = JointConstraint()

            constraint.joint_name = name
            constraint.position = position
            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0

            constraints.joint_constraints.append(constraint)

        goal.request.goal_constraints.append(constraints)

        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.2
        goal.request.max_acceleration_scaling_factor = 0.2

        self.get_logger().info(
            'Sending joint-space goal to MoveIt...'
        )

        future = self.action_client.send_goal_async(
            goal
        )

        future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                'MoveIt rejected the goal'
            )
            return

        self.get_logger().info(
            'Goal accepted by MoveIt'
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.result_callback
        )

    def result_callback(self, future):

        result = future.result()

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                'UR10 motion completed successfully!'
            )
        else:
            self.get_logger().error(
                f'Motion failed. Status: {result.status}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = UR10JointControl()

    # Example joint configuration
    target = [
        0.0,
        -1.0,
        1.0,
        -1.5,
        -1.57,
        0.0
    ]

    node.move_to_joint_position(target)

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()