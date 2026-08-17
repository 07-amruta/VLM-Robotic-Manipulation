import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from moveit_msgs.action import MoveGroup, ExecuteTrajectory


class UR10MultiMotionControl(Node):

    def __init__(self):
        super().__init__('ur10_multi_motion_control')

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

        # Target Configurations (in radians)
        self.POSITIONS = {
            'home': [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            'test_configuration': [0.0, -1.2, 1.2, -1.57, -1.57, 0.0],
            'up': [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # Fully extended vertical pose
        }

        # Action & Service Clients
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')

        # Joint State Subscriber
        self.current_joint_state = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Motion Parameters
        self.eef_step = 0.005          # 5mm steps for higher precision
        self.jump_threshold = 0.0      # Disable jump threshold check
        self.minimum_fraction = 0.90   # Require 90% completion

        # Sequence order: Perform Cartesian path while in "test_configuration" pose
        self.sequence_step = 0
        self.motion_sequence = ['home', 'test_configuration', 'cartesian_path', 'up']

        self.get_logger().info('Waiting for MoveIt 2 interfaces...')
        self.move_group_client.wait_for_server()
        self.cartesian_path_client.wait_for_service()
        self.execute_trajectory_client.wait_for_server()
        self.fk_client.wait_for_service()

        self.get_logger().info('MoveIt 2 interfaces ready.')

    def joint_state_callback(self, msg):
        if all(name in msg.name for name in self.joint_names):
            ordered_msg = JointState()
            ordered_msg.header = msg.header
            ordered_msg.name = self.joint_names
            ordered_msg.position = [msg.position[msg.name.index(n)] for n in self.joint_names]
            self.current_joint_state = ordered_msg

    def execute_next_step(self):
        if self.sequence_step >= len(self.motion_sequence):
            self.get_logger().info('============================================')
            self.get_logger().info('ALL SEQUENCE MOTIONS COMPLETED SUCCESSFULLY')
            self.get_logger().info('============================================')
            rclpy.shutdown()

            if sequence_finished:
                self.get_logger().info("Sequence complete. Exiting...")
                self.destroy_node()
                raise SystemExit  # Stops rclpy.spin() cleanly

            return

        current_target = self.motion_sequence[self.sequence_step]
        self.get_logger().info(f'---> Step {self.sequence_step + 1}/{len(self.motion_sequence)}: Moving to "{current_target}"')

        if current_target in self.POSITIONS:
            self.move_to_joint_target(self.POSITIONS[current_target])
        elif current_target == 'cartesian_path':
            self.get_current_ee_pose_and_plan()

    def move_to_joint_target(self, joint_positions):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = 0.4
        goal.request.max_acceleration_scaling_factor = 0.4

        from moveit_msgs.msg import Constraints, JointConstraint
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
            rclpy.shutdown()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.joint_result_callback)

    def joint_result_callback(self, future):
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().error(f'Joint execution failed code: {result.error_code.val}')
            rclpy.shutdown()
            return

        self.get_logger().info(f'Reached target configuration: {self.motion_sequence[self.sequence_step]}')
        self.sequence_step += 1

        # Brief pause between steps
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.timer.cancel()
        self.execute_next_step()

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

        # 15 cm x 15 cm Reachable Cartesian Square Pattern
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
            rclpy.shutdown()
            return

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        future_exec = self.execute_trajectory_client.send_goal_async(goal)
        future_exec.add_done_callback(self.execute_goal_response)

    def execute_goal_response(self, future):
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.cartesian_result_callback)

    def cartesian_result_callback(self, future):
        self.get_logger().info('Completed Cartesian Path Step.')
        self.sequence_step += 1
        self.execute_next_step()


def main(args=None):
    rclpy.init(args=args)
    node = UR10MultiMotionControl()
    node.execute_next_step()
    rclpy.spin(node)

    node = CartesianTrajectoryControl()
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()