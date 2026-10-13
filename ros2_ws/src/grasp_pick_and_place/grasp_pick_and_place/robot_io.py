import time
from collections import Counter

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseArray
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import (
    ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetPositionIK, GetStateValidity)
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from tf2_ros import Buffer, TransformListener

from grasp_pick_and_place.config import ARM_JOINTS, GRIPPER_JOINTS, HARDWARE_MODE, PARAMS


class RobotIO(Node):
    """Parameters, robot/scene state, TF, MoveIt and gripper clients, blocking helpers."""

    def __init__(self):
        super().__init__('pick_executor')

        for name, value in PARAMS.items():
            self.declare_parameter(name, value)

        self.candidates = None
        self.scene_points = None
        self.scene_frame = None
        self.joint_positions = None
        self.finger_positions = None
        self.joint_map = {}
        self.gripper_links = []
        self.rejections = Counter()

        self.frame_id = None
        self.tcp_to_link = None
        self.hand_axis = None      # closing axis of the open hand, unit, tcp frame
        self.hand_centre = None    # centre of the closing region of the open hand, tcp frame
        self.object_centre = None
        self.object_size = None
        self.box_axes = None       # +-x and +-y of the object box, in the scene frame
        self.round_object = False
        self.sphere = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(PoseArray, 'grasp_candidates', self.on_candidates, latched)
        self.create_subscription(PointCloud2, 'scene_points', self.on_cloud, latched)
        self.create_subscription(JointState, '/joint_states', self.on_joint_state, 10)

        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        # The real gripper is driven over Modbus, not through a controller
        if HARDWARE_MODE != 'real':
            self.gripper_client = ActionClient(
                self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.get_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.apply_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.description_client = self.create_client(
            GetParameters, '/move_group/get_parameters')
        self.validity_client = self.create_client(GetStateValidity, '/check_state_validity')

        self.get_logger().info('Waiting for MoveIt 2 and gripper interfaces...')
        self.move_group_client.wait_for_server()
        self.execute_client.wait_for_server()
        if HARDWARE_MODE != 'real':
            self.gripper_client.wait_for_server()
        self.ik_client.wait_for_service()
        self.cartesian_client.wait_for_service()
        self.get_scene_client.wait_for_service()
        self.apply_scene_client.wait_for_service()
        self.validity_client.wait_for_service()
        self.get_logger().info('Interfaces ready.')

    # ---------- subscriptions ----------

    def on_candidates(self, msg):
        self.candidates = msg

    def on_cloud(self, msg):
        self.scene_points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
        self.scene_frame = msg.header.frame_id

    def on_joint_state(self, msg):
        self.joint_map.update(zip(msg.name, msg.position))
        # Look joints up by name; JointState does not guarantee ordering
        if all(n in msg.name for n in ARM_JOINTS):
            self.joint_positions = [msg.position[msg.name.index(n)] for n in ARM_JOINTS]
        if all(n in msg.name for n in GRIPPER_JOINTS):
            self.finger_positions = [msg.position[msg.name.index(n)] for n in GRIPPER_JOINTS]

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

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def ready(self):
        return (
            self.candidates is not None and self.candidates.poses
            and self.scene_points is not None
            and self.joint_positions is not None
            and self.finger_positions is not None
            and self.tf_buffer.can_transform(
                self.get_parameter('ik_link').value,
                self.get_parameter('tcp_frame').value, Time())
        )