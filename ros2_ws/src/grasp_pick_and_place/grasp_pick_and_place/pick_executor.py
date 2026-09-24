import sys

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from grasp_pick_and_place.transform_utils import lookup_matrix, matrix_to_pose, pose_to_matrix

ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
GRIPPER_JOINTS = [
    'gripper_finger_1_joint_1', 'gripper_finger_2_joint_1', 'gripper_finger_middle_joint_1',
]
MOVEIT_SUCCESS = 1


class PickExecutor(Node):
    """Runs one pick: open, pre-grasp, grasp, close, lift, up.

    Every arm move is target pose -> IK (/compute_ik) -> joint-space goal (/move_action).
    """

    def __init__(self):
        super().__init__('pick_executor')

        self.declare_parameter('group_name', 'ur_manipulator')
        self.declare_parameter('ik_link', 'tool0')
        self.declare_parameter('tcp_frame', 'gripper_tool0')
        self.declare_parameter('approach_distance', 0.10)
        self.declare_parameter('lift_height', 0.15)
        self.declare_parameter('velocity_scaling', 0.4)
        self.declare_parameter('gripper_open', [0.05, 0.05, 0.05])
        self.declare_parameter('gripper_close', [0.7, 0.7, 0.7])
        self.declare_parameter('gripper_duration', 1.5)
        self.declare_parameter('up_joints', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0])

        self.grasp_pose = None
        self.joint_positions = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(PoseStamped, 'grasp_pose', self.on_grasp_pose, latched)
        self.create_subscription(JointState, '/joint_states', self.on_joint_state, 10)

        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        self.get_logger().info('Waiting for MoveIt 2 and gripper interfaces...')
        self.move_group_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.ik_client.wait_for_service()
        self.get_logger().info('Interfaces ready.')

    def on_grasp_pose(self, msg):
        self.grasp_pose = msg

    def on_joint_state(self, msg):
        # Look joints up by name; JointState does not guarantee ordering
        if all(n in msg.name for n in ARM_JOINTS):
            self.joint_positions = [msg.position[msg.name.index(n)] for n in ARM_JOINTS]

    # ---------- blocking helpers ----------

    def call_service(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def run_goal(self, client, goal):
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if not handle.accepted:
            return None
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result().result

    # ---------- IK and motion ----------

    def solve_ik(self, pose_matrix, frame_id, seed):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self.get_parameter('group_name').value
        ik.ik_link_name = self.get_parameter('ik_link').value
        ik.avoid_collisions = True
        ik.timeout = Duration(sec=1)
        ik.robot_state.joint_state = JointState(
            name=ARM_JOINTS, position=[float(p) for p in seed])
        ik.pose_stamped.header.frame_id = frame_id
        ik.pose_stamped.pose = matrix_to_pose(pose_matrix)

        response = self.call_service(self.ik_client, request)
        if response.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().error(f'IK failed (code {response.error_code.val})')
            return None
        solved = dict(zip(response.solution.joint_state.name,
                          response.solution.joint_state.position))
        return [solved[n] for n in ARM_JOINTS]

    def move_to_joints(self, positions):
        scaling = self.get_parameter('velocity_scaling').value
        goal = MoveGroup.Goal()
        goal.request.group_name = self.get_parameter('group_name').value
        goal.request.max_velocity_scaling_factor = scaling
        goal.request.max_acceleration_scaling_factor = scaling

        constraints = Constraints()
        for name, position in zip(ARM_JOINTS, positions):
            constraints.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(position),
                tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = False

        result = self.run_goal(self.move_group_client, goal)
        if result is None or result.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().error('Joint move failed or was rejected')
            return False
        return True

    def move_gripper(self, positions):
        duration = self.get_parameter('gripper_duration').value
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINTS
        goal.trajectory.points = [point]

        result = self.run_goal(self.gripper_client, goal)
        if result is None:
            self.get_logger().error('Gripper goal rejected')
            return False
        # Fingers stalling on the object can report a tolerance error; that is expected when closing
        self.get_logger().info(f'Gripper finished (error_code {result.error_code})')
        return True

    # ---------- sequence ----------

    def ready(self):
        return (
            self.grasp_pose is not None
            and self.joint_positions is not None
            and self.tf_buffer.can_transform(
                self.get_parameter('ik_link').value,
                self.get_parameter('tcp_frame').value, Time())
        )

    def run(self):
        self.get_logger().info('Waiting for /grasp_pose, /joint_states and TF...')
        while rclpy.ok() and not self.ready():
            rclpy.spin_once(self, timeout_sec=0.1)

        frame_id = self.grasp_pose.header.frame_id
        grasp = pose_to_matrix(self.grasp_pose.pose)
        try:
            link_from_tcp = lookup_matrix(
                self.tf_buffer, self.get_parameter('ik_link').value,
                self.get_parameter('tcp_frame').value)
        except TransformException as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return False
        tcp_to_link = np.linalg.inv(link_from_tcp)

        pre = grasp.copy()
        pre[:3, 3] -= self.get_parameter('approach_distance').value * grasp[:3, 0]
        lift = grasp.copy()
        lift[2, 3] += self.get_parameter('lift_height').value

        # Solve every IK target before moving anything, each seeded with the previous
        # solution to stay on the same IK branch
        seed = self.joint_positions
        solutions = []
        for name, target in (('pre-grasp', pre), ('grasp', grasp), ('lift', lift)):
            joints = self.solve_ik(target @ tcp_to_link, frame_id, seed)
            if joints is None:
                self.get_logger().error(f'No IK solution for {name}; nothing was moved')
                return False
            solutions.append(joints)
            seed = joints
        pre_joints, grasp_joints, lift_joints = solutions

        open_pos = self.get_parameter('gripper_open').value
        close_pos = self.get_parameter('gripper_close').value
        up_joints = self.get_parameter('up_joints').value
        steps = [
            ('open gripper', lambda: self.move_gripper(open_pos)),
            ('pre-grasp', lambda: self.move_to_joints(pre_joints)),
            ('grasp', lambda: self.move_to_joints(grasp_joints)),
            ('close gripper', lambda: self.move_gripper(close_pos)),
            ('lift', lambda: self.move_to_joints(lift_joints)),
            ('up', lambda: self.move_to_joints(up_joints)),
        ]
        for i, (name, action) in enumerate(steps, 1):
            self.get_logger().info(f'Step {i}/{len(steps)}: {name}')
            if not action():
                self.get_logger().error(f'Step failed: {name}')
                return False
        self.get_logger().info('Pick complete')
        return True


def main():
    rclpy.init()
    node = PickExecutor()
    success = False
    try:
        success = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if success else 1)