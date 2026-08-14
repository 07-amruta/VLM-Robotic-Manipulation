import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint


class UR10JointControl(Node):

    def __init__(self):
        super().__init__('ur10_joint_control')

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

        self.get_logger().info('Waiting for MoveIt 2...')

        self.move_group_client.wait_for_server()

        self.get_logger().info(
            'MoveIt 2 MoveGroup action connected.'
        )

    def create_goal(self, joint_positions):

        goal = MoveGroup.Goal()

        # --------------------------------------------------
        # Planning request
        # --------------------------------------------------

        goal.request.group_name = 'ur_manipulator'

        # Slow down the motion
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1

        # --------------------------------------------------
        # Joint constraints
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Planning options
        # --------------------------------------------------

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True

        return goal

    def move_to_position(self, joint_positions, description):

        self.get_logger().info(
            f'Moving to {description}...'
        )

        goal = self.create_goal(joint_positions)

        send_goal_future = (
            self.move_group_client.send_goal_async(goal)
        )

        send_goal_future.add_done_callback(
            lambda future: self.goal_response_callback(
                future,
                description
            )
        )

    def goal_response_callback(self, future, description):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.get_logger().error(
                f'Move to {description} was rejected.'
            )

            rclpy.shutdown()
            return

        self.get_logger().info(
            f'{description} goal accepted.'
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            lambda future: self.result_callback(
                future,
                description
            )
        )

    def result_callback(self, future, description):

        result = future.result().result

        error_code = result.error_code.val

        if error_code == 1:

            self.get_logger().info(
                f'{description} reached successfully!'
            )

            # ----------------------------------------------
            # START → TARGET
            # ----------------------------------------------

            if description == 'START POSITION':

                self.get_logger().info(
                    'Starting movement to TARGET POSITION...'
                )

                self.move_to_position(
                    self.target_position,
                    'TARGET POSITION'
                )

            else:

                self.get_logger().info(
                    'Motion sequence completed!'
                )

                rclpy.shutdown()

        else:

            self.get_logger().error(
                f'Motion to {description} failed. '
                f'MoveIt error code: {error_code}'
            )

            rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = UR10JointControl()

    # ======================================================
    # START POSITION
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
    # TARGET POSITION
    # ======================================================

    node.target_position = [
        0.0,
        -1.0,
        1.0,
        -1.5,
        -1.57,
        0.0
    ]

    # ======================================================
    # START SEQUENCE
    # ======================================================

    node.get_logger().info(
        '========================================'
    )

    node.get_logger().info(
        'UR10 JOINT MOTION SEQUENCE'
    )

    node.get_logger().info(
        '========================================'
    )

    node.get_logger().info(
        'Step 1: Move to START POSITION'
    )

    node.get_logger().info(
        'Step 2: Move to TARGET POSITION'
    )

    node.move_to_position(
        node.start_position,
        'START POSITION'
    )

    rclpy.spin(node)


if __name__ == '__main__':
    main()