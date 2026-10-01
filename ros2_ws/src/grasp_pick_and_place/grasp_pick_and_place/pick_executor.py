import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry, AllowedCollisionMatrix, CollisionObject, Constraints,
    JointConstraint, PlanningScene, PlanningSceneComponents, RobotState)
from moveit_msgs.srv import (
    ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetPositionIK, GetStateValidity)
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState, PointCloud2
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
from scipy.spatial import ConvexHull

from grasp_pick_and_place.transform_utils import lookup_matrix, matrix_to_pose, pose_to_matrix

ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
GRIPPER_JOINTS = [
    'gripper_finger_1_joint_1', 'gripper_finger_2_joint_1', 'gripper_finger_middle_joint_1',
]
MOVEIT_SUCCESS = 1
OBJECT_ID = 'target_object'
SUPPORT_ID = 'support_table'


def fit_box(points, margin):
    """Yaw-aligned box around a cloud: minimum-area rectangle in xy, full extent in z.

    Returns (centre, size, yaw). The bottom face sits on the cloud's lowest point (the table).
    """
    xy = points[:, :2]
    best = None
    for yaw in np.deg2rad(np.arange(0.0, 90.0, 1.0)):
        c, s = np.cos(yaw), np.sin(yaw)
        local = xy @ np.array([[c, -s], [s, c]])  # coordinates along (c, s) and (-s, c)
        low, high = local.min(axis=0), local.max(axis=0)
        area = np.prod(high - low)
        if best is None or area < best[0] * 0.98:
            best = (area, yaw, low, high)
    _, yaw, low, high = best
    fill = ConvexHull(xy).volume / np.prod(high - low)
    c, s = np.cos(yaw), np.sin(yaw)
    centre_xy = np.array([[c, -s], [s, c]]) @ ((low + high) / 2.0)
    z_low, z_high = points[:, 2].min(), points[:, 2].max()
    centre = np.array([centre_xy[0], centre_xy[1], (z_low + z_high + margin) / 2.0])
    size = np.array([*(high - low + 2.0 * margin), z_high - z_low + margin])
    return centre, size, yaw, fill


def is_similar(a, b, max_dist=0.01, max_angle=0.1):
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return np.linalg.norm(a[:3, 3] - b[:3, 3]) < max_dist and np.arccos(cos) < max_angle

def longest_run(flags):
    """(first, last) indices of the longest run of True values, or None."""
    best, start = None, None
    for k, ok in enumerate(list(flags) + [False]):
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            if best is None or k - 1 - start > best[1] - best[0]:
                best = (start, k - 1)
            start = None
    return best

class PickExecutor(Node):
    """Picks one object: open, pre-grasp, straight approach, close, straight lift, up.

    Every proposed grasp is first turned into a hand pose that envelopes the object (closing
    axis on a box face normal, hand centred on the object). A hand pose is only used if the
    open hand touches nothing there, and if IK, the pre-grasp plan, the straight approach,
    the lift and the up move are all feasible. Nothing moves until a pose passes every check.
    """

    def __init__(self):
        super().__init__('pick_executor')

        self.declare_parameter('group_name', 'ur_manipulator')
        self.declare_parameter('ik_link', 'tool0')
        self.declare_parameter('tcp_frame', 'gripper_tool0')
        self.declare_parameter('approach_margin', 0.05)
        self.declare_parameter('lift_height', 0.15)
        self.declare_parameter('up_joints', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        self.declare_parameter('velocity_scaling', 0.4)
        self.declare_parameter('planning_time', 3.0)
        self.declare_parameter('ik_seeds', 8)
        self.declare_parameter('ik_timeout', 0.1)
        self.declare_parameter('min_path_fraction', 0.99)
        self.declare_parameter('cartesian_step', 0.01)
        self.declare_parameter('cartesian_joint_speed', 0.3)
        self.declare_parameter('max_joint_step', 0.3)
        self.declare_parameter('object_margin', 0.01)
        self.declare_parameter('gripper_link_prefix', 'gripper_')
        self.declare_parameter('gripper_links', '')  # comma-separated override
        self.declare_parameter('gripper_open', [0.05, 0.05, 0.05])
        self.declare_parameter('gripper_duration', 1.5)
        self.declare_parameter('close_step', 0.03)
        self.declare_parameter('close_period', 0.1)
        self.declare_parameter('close_limit', [1.2, 1.2, 1.2])
        self.declare_parameter('contact_lag', 0.08)
        self.declare_parameter('squeeze_margin', 0.01)
        self.declare_parameter('close_timeout', 15.0)
        self.declare_parameter('motion_eps', 0.01)
        self.declare_parameter('stall_count', 2)
        self.declare_parameter(
            'hand_pair_frames', ['gripper_finger_1_link_3', 'gripper_finger_2_link_3'])
        self.declare_parameter('hand_opposing_frame', 'gripper_finger_middle_link_3')
        self.declare_parameter('hand_palm_frame', 'gripper_palm')
        self.declare_parameter('search_step', 0.005)
        self.declare_parameter('search_reach', 0.04)
        self.declare_parameter('retreat_step', 0.01)
        self.declare_parameter('max_retreat', 0.03)
        self.declare_parameter('min_free_width', 0.01)
        self.declare_parameter('fallback_depth', 0.02)
        self.declare_parameter('support_centre', [0.6, 0.0])  # xy of the table block, scene frame
        self.declare_parameter('support_size', [0.6, 0.8])    # xy extent of the table top
        self.declare_parameter('prismatic_fill', 0.9)

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

    # ---------- planning scene ----------

    def apply_scene(self, scene):
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self.call_service(self.apply_scene_client, request)
        if response is None or not response.success:
            raise RuntimeError('Planning scene update was rejected')

    def discover_gripper_links(self):
        override = [n.strip() for n in self.get_parameter('gripper_links').value.split(',')
                    if n.strip()]
        if override:
            return override
        if not self.description_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                'Cannot read robot_description from /move_group; set the gripper_links parameter')
        request = GetParameters.Request()
        request.names = ['robot_description']
        response = self.call_service(self.description_client, request)
        urdf = response.values[0].string_value
        prefix = self.get_parameter('gripper_link_prefix').value
        links = [link.get('name') for link in ET.fromstring(urdf).iter('link')]
        found = [name for name in links if name.startswith(prefix)]
        if not found:
            raise RuntimeError(f'No links start with "{prefix}"; set the gripper_links parameter')
        return found

    def remove_object(self):
        """Clears a target_object left over from an earlier run."""
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        current = self.call_service(self.get_scene_client, request).scene
        if any(o.id == OBJECT_ID for o in current.world.collision_objects):
            remove = CollisionObject()
            remove.id = OBJECT_ID
            remove.operation = CollisionObject.REMOVE
            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects = [remove]
            self.apply_scene(scene)

    def add_object(self):
        """Fits a yaw-aligned box to the scene cloud and adds it to MoveIt's scene."""
        points = self.scene_points.astype(float)
        if points.shape[0] < 20:
            raise RuntimeError('Too few cloud points to model the object')
        centre, size, yaw, fill = fit_box(points, self.get_parameter('object_margin').value)
        self.round_object = fill < self.get_parameter('prismatic_fill').value
        self.object_centre, self.object_size = centre, size
        ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ey = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        self.box_axes = (ex, -ex, ey, -ey)

        obj = CollisionObject()
        obj.header.frame_id = self.scene_frame
        obj.id = OBJECT_ID
        primitive = SolidPrimitive()
        if self.round_object:
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [float(size[2]), float(size[:2].mean() / 2.0)]  # height, radius
        else:
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [float(v) for v in size]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, centre)
        qx, qy, qz, qw = Rotation.from_euler('z', yaw).as_quat()
        pose.orientation.x, pose.orientation.y = float(qx), float(qy)
        pose.orientation.z, pose.orientation.w = float(qz), float(qw)
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        self.apply_scene(scene)
        self.get_logger().info(
            f'Object box size {np.round(size, 3).tolist()} centre {np.round(centre, 3).tolist()} '
            f'yaw {np.rad2deg(yaw):.0f} deg'
            f'footprint fill {fill:.2f}')

    def add_support(self):
        """Adds the table as a block on the floor whose top is the cloud's lowest point."""
        top = self.object_centre[2] - self.object_size[2] / 2.0
        size_x, size_y = self.get_parameter('support_size').value
        centre_x, centre_y = self.get_parameter('support_centre').value
        obj = CollisionObject()
        obj.header.frame_id = self.scene_frame
        obj.id = SUPPORT_ID
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(size_x), float(size_y), float(top)]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(centre_x), float(centre_y), float(top / 2.0)
        pose.orientation.w = 1.0
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        self.apply_scene(scene)

    def set_contact_allowed(self, allowed):
        """Allow or forbid gripper-object contact in MoveIt's collision matrix."""
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        acm = self.call_service(
            self.get_scene_client, request).scene.allowed_collision_matrix

        names = list(acm.entry_names)
        rows = [list(entry.enabled) for entry in acm.entry_values]
        if OBJECT_ID not in names:
            for row in rows:
                row.append(False)
            names.append(OBJECT_ID)
            rows.append([False] * len(names))
        k = names.index(OBJECT_ID)
        for i, name in enumerate(names):
            if name in self.gripper_links:
                rows[k][i] = allowed
                rows[i][k] = allowed

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = AllowedCollisionMatrix(
            entry_names=names,
            entry_values=[AllowedCollisionEntry(enabled=row) for row in rows],
            default_entry_names=list(acm.default_entry_names),
            default_entry_values=list(acm.default_entry_values))
        self.apply_scene(scene)

    # ---------- IK and motion ----------

    def solve_ik(self, tcp_pose, seed, avoid_collisions=True):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self.get_parameter('group_name').value
        ik.ik_link_name = self.get_parameter('ik_link').value
        ik.avoid_collisions = avoid_collisions
        timeout = self.get_parameter('ik_timeout').value
        ik.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1) * 1e9))
        ik.robot_state.joint_state = JointState(
            name=ARM_JOINTS, position=[float(p) for p in seed])
        ik.pose_stamped.header.frame_id = self.frame_id
        ik.pose_stamped.pose = matrix_to_pose(tcp_pose @ self.tcp_to_link)

        response = self.call_service(self.ik_client, request)
        if response.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().debug(f'IK failed (code {response.error_code.val})')
            return None
        solved = dict(zip(response.solution.joint_state.name,
                          response.solution.joint_state.position))
        joints = [solved[n] for n in ARM_JOINTS]
        # Use the angle equivalent (mod a full turn) that is nearest the current joints
        return [c + (q - c + np.pi) % (2 * np.pi) - np.pi
                for q, c in zip(joints, self.joint_positions)]

    def joint_goal(self, positions):
        scaling = self.get_parameter('velocity_scaling').value
        goal = MoveGroup.Goal()
        goal.request.group_name = self.get_parameter('group_name').value
        goal.request.max_velocity_scaling_factor = scaling
        goal.request.max_acceleration_scaling_factor = scaling
        goal.request.allowed_planning_time = self.get_parameter('planning_time').value

        constraints = Constraints()
        for name, position in zip(ARM_JOINTS, positions):
            constraints.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(position),
                tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
        goal.request.goal_constraints.append(constraints)
        return goal

    def move_to_joints(self, positions):
        """Plan and execute a joint-space move from the current state."""
        goal = self.joint_goal(positions)
        goal.planning_options.plan_only = False
        result = self.run_goal(self.move_group_client, goal)
        if result is None:
            self.get_logger().error('Joint move rejected by move_group')
            return False
        if result.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().error(
                f'Joint move failed (MoveIt error code {result.error_code.val})')
            return False
        return True

    def plan_to_joints(self, start_joints, goal_joints):
        """Plan only, from a given start state. Returns a RobotTrajectory or None."""
        goal = self.joint_goal(goal_joints)
        goal.request.start_state.joint_state = JointState(
            name=ARM_JOINTS, position=[float(p) for p in start_joints])
        goal.planning_options.plan_only = True
        result = self.run_goal(self.move_group_client, goal)
        if result is None or result.error_code.val != MOVEIT_SUCCESS:
            return None
        return result.planned_trajectory

    def time_parameterize(self, trajectory):
        """Positions-only timing: constant joint speed, strictly increasing stamps."""
        speed = self.get_parameter('cartesian_joint_speed').value
        elapsed, previous = 0.0, None
        for point in trajectory.joint_trajectory.points:
            if previous is not None:
                delta = max(abs(a - b) for a, b in zip(point.positions, previous))
                elapsed += max(delta / speed, 0.01)
            previous = list(point.positions)
            sec = int(elapsed)
            point.time_from_start = Duration(sec=sec, nanosec=int((elapsed - sec) * 1e9))
            point.velocities = []
            point.accelerations = []

    def plan_line(self, start_joints, target):
        """Straight-line tool path to the tcp pose `target`. Returns (trajectory, end) or None."""
        request = GetCartesianPath.Request()
        request.header.frame_id = self.frame_id
        request.group_name = self.get_parameter('group_name').value
        request.link_name = self.get_parameter('ik_link').value
        request.start_state = RobotState()
        request.start_state.joint_state = JointState(
            name=ARM_JOINTS, position=[float(p) for p in start_joints])
        request.waypoints = [matrix_to_pose(target @ self.tcp_to_link)]
        request.max_step = self.get_parameter('cartesian_step').value
        request.jump_threshold = 0.0
        request.avoid_collisions = True

        response = self.call_service(self.cartesian_client, request)
        fraction = response.fraction
        if (response.error_code.val != MOVEIT_SUCCESS
                or fraction < self.get_parameter('min_path_fraction').value):
            self.get_logger().warn(f'Straight-line path only {fraction:.0%} feasible')
            return None

        trajectory = response.solution
        points = trajectory.joint_trajectory.points
        if len(points) < 2:
            return None
        worst = max(max(abs(a - b) for a, b in zip(p.positions, q.positions))
                    for p, q in zip(points[1:], points[:-1]))
        if worst > self.get_parameter('max_joint_step').value:
            self.get_logger().warn(f'Straight-line path has a joint jump of {worst:.2f} rad')
            return None

        names = trajectory.joint_trajectory.joint_names
        end = [points[-1].positions[names.index(n)] for n in ARM_JOINTS]
        self.time_parameterize(trajectory)
        return trajectory, end

    def execute_trajectory(self, trajectory):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        result = self.run_goal(self.execute_client, goal)
        if result is None or result.error_code.val != MOVEIT_SUCCESS:
            code = None if result is None else result.error_code.val
            self.get_logger().error(f'Trajectory execution failed (code {code})')
            return False
        return True

    def line_move(self, target, label):
        """Straight-line move from the actual current state; raises if it fails."""
        self.spin_for(0.3)  # let /joint_states settle before planning from the current state
        plan = self.plan_line(self.joint_positions, target)
        if plan is None or not self.execute_trajectory(plan[0]):
            raise RuntimeError(f'{label} failed')

    # ---------- gripper ----------

    def finger_goal(self, positions, duration):
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINTS
        goal.trajectory.points = [point]
        return goal

    def move_gripper(self, positions):
        goal = self.finger_goal(positions, self.get_parameter('gripper_duration').value)
        result = self.run_goal(self.gripper_client, goal)
        if result is None:
            self.get_logger().error('Gripper goal rejected')
            return False
        # Fingers stalling on the object can report a tolerance error; that is expected when closing
        self.get_logger().info(f'Gripper finished (error_code {result.error_code})')
        return True

    def close_until_contact(self):
        step = self.get_parameter('close_step').value
        period = self.get_parameter('close_period').value
        limits = list(self.get_parameter('close_limit').value)
        lag = self.get_parameter('contact_lag').value
        margin = self.get_parameter('squeeze_margin').value
        eps = self.get_parameter('motion_eps').value
        needed = self.get_parameter('stall_count').value
        deadline = time.monotonic() + self.get_parameter('close_timeout').value

        start = list(self.finger_positions)
        target = list(start)
        previous = list(start)
        still = [0] * len(GRIPPER_JOINTS)
        stalled = [False] * len(GRIPPER_JOINTS)
        while rclpy.ok() and time.monotonic() < deadline:
            for i, name in enumerate(GRIPPER_JOINTS):
                if stalled[i]:
                    continue
                measured = self.finger_positions[i]
                still[i] = still[i] + 1 if abs(measured - previous[i]) < eps else 0
                previous[i] = measured
                has_moved = measured - start[i] > step
                blocked = has_moved and still[i] >= needed and target[i] - measured > lag
                if blocked:
                    stalled[i] = True
                    target[i] = measured + margin
                    self.get_logger().info(f'{name} stalled at {measured:.3f}')
                elif target[i] >= limits[i]:
                    self.get_logger().error(f'{name} reached its limit without contact')
                    return False
                else:
                    target[i] = min(target[i] + step, limits[i])

            self.gripper_client.send_goal_async(self.finger_goal(target, period))
            self.spin_for(period)
            if all(stalled):
                return True
        self.get_logger().error('Gripper close timed out')
        return False

    # ---------- hand and grasp geometry ----------

    def hand_geometry(self):
        """Closing axis and closing centre of the open hand in the tcp frame, from TF."""
        tcp = self.get_parameter('tcp_frame').value

        def position(frame):
            return lookup_matrix(self.tf_buffer, tcp, frame)[:3, 3]

        palm = position(self.get_parameter('hand_palm_frame').value)
        if palm[0] >= 0.0:
            raise RuntimeError(
                f'Palm is at x={palm[0]:.3f} in the tcp frame; the grasp frame assumes x points '
                'from the palm toward the fingertips')
        pair = np.mean([position(f) for f in self.get_parameter('hand_pair_frames').value],
                       axis=0)
        opposing = position(self.get_parameter('hand_opposing_frame').value)
        centre = (pair + opposing) / 2.0
        direction = pair - opposing
        centre[0] = direction[0] = 0.0  # x is the approach axis of the grasp frame
        gap = float(np.linalg.norm(direction))
        self.hand_axis = direction / gap
        self.hand_centre = centre
        self.get_logger().info(
            f'Hand closing axis {np.round(self.hand_axis, 3).tolist()} '
            f'centre {np.round(self.hand_centre, 3).tolist()} gap {gap:.3f} m, '
            f'palm x {palm[0]:.3f} (tcp frame)')

    def hand_poses(self, proposal):
        """Turns a proposed grasp into hand poses that envelope the object.

        The hand's closing axis is put on the proposal's jaw axis, then snapped to the nearest
        box face normal, and the hand's closing centre is moved onto the object's centre
        across the approach. Approach axis and depth stay as proposed. Both 180-degree wrist
        variants are returned.
        """
        approach = proposal[:3, 0]
        tilt = -np.arctan2(self.hand_axis[2], self.hand_axis[1])
        rot = proposal[:3, :3] @ Rotation.from_rotvec([tilt, 0.0, 0.0]).as_matrix()
        closing = rot @ self.hand_axis

        turns = []
        for axis in self.box_axes:
            target = axis - approach * (axis @ approach)
            target /= np.linalg.norm(target)
            turns.append(np.arctan2(approach @ np.cross(closing, target), closing @ target))

        snap = min(turns, key=abs)
        if self.round_object:
            # Every horizontal direction is a "face" of an upright round object: close level
            level = np.cross([0.0, 0.0, 1.0], approach)  # horizontal and perpendicular to the approach
            if np.linalg.norm(level) > 1e-6:  # a vertical approach keeps the box-face snap
                level /= np.linalg.norm(level)
                level *= np.sign(closing @ level) or 1.0  # keep the smaller of the two turns
                snap = np.arctan2(approach @ np.cross(closing, level), closing @ level)

        offset = self.object_centre - proposal[:3, 3]
        offset -= approach * (offset @ approach)
        poses = []
        for extra in (0.0, np.pi):
            r = Rotation.from_rotvec(approach * (snap + extra)).as_matrix() @ rot
            pose = np.eye(4)
            pose[:3, :3] = r
            pose[:3, 3] = proposal[:3, 3] + offset - r @ self.hand_centre
            poses.append(pose)
        return poses

    def top_down_grasp(self):
        """Fallback proposal: straight down onto the object, closing across its narrower side."""
        ex, _, ey, _ = self.box_axes
        closing = ex if self.object_size[0] <= self.object_size[1] else ey
        approach = np.array([0.0, 0.0, -1.0])
        top = self.object_centre[2] + self.object_size[2] / 2.0
        grasp = np.eye(4)
        grasp[:3, :3] = np.column_stack([approach, closing, np.cross(approach, closing)])
        grasp[:3, 3] = [
            self.object_centre[0], self.object_centre[1],
            top - self.get_parameter('fallback_depth').value]
        return grasp

    def object_contacts(self, joints):
        """Names of the robot bodies touching the object in this arm configuration."""
        positions = dict(self.joint_map)
        positions.update(zip(ARM_JOINTS, (float(q) for q in joints)))
        request = GetStateValidity.Request()
        request.group_name = self.get_parameter('group_name').value
        request.robot_state.joint_state = JointState(
            name=list(positions), position=list(positions.values()))
        response = self.call_service(self.validity_client, request)
        touched = (OBJECT_ID, SUPPORT_ID)
        return sorted({c.contact_body_2 if c.contact_body_1 in touched else c.contact_body_1
                       for c in response.contacts
                       if c.contact_body_1 in touched or c.contact_body_2 in touched})

    def find_envelope(self, grasp):
        """Slides the open hand across its closing axis, backing off along the approach if
        needed, to where it touches nothing. Returns (pose, free_width, retreat) or None."""
        step = self.get_parameter('search_step').value
        reach = self.get_parameter('search_reach').value
        offsets = np.arange(-reach, reach + 1e-9, step)
        retreats = np.arange(0.0, self.get_parameter('max_retreat').value + 1e-9,
                             self.get_parameter('retreat_step').value)
        closing = grasp[:3, :3] @ self.hand_axis
        approach = grasp[:3, 0]
        seed = list(self.joint_positions)
        closest = None
        for retreat in retreats:
            free = []
            for offset in offsets:
                moved = grasp.copy()
                moved[:3, 3] += closing * offset - approach * retreat
                joints = self.solve_ik(moved, seed, avoid_collisions=False)
                if joints is None:
                    free.append(False)
                    continue
                contacts = self.object_contacts(joints)
                free.append(not contacts)
                if closest is None or len(contacts) < len(closest[0]):
                    closest = (contacts, offset, retreat)
            run = longest_run(free)
            if run is not None and (run[1] - run[0] + 1) * step >= self.get_parameter(
                    'min_free_width').value:
                moved = grasp.copy()
                moved[:3, 3] += closing * (offsets[run[0]] + offsets[run[1]]) / 2.0 \
                    - approach * retreat
                return moved, (run[1] - run[0] + 1) * step, retreat
        if closest is None:
            self.get_logger().info('Closest miss: no reachable hand position')
        else:
            contacts, offset, retreat = closest
            self.get_logger().info(
                f'Closest miss: {contacts} at offset {offset:+.3f} m, retreat {retreat:.3f} m')
        return None

    # ---------- one hand pose: check, then execute ----------

    def targets(self, grasp):
        # Start outside the object's bounding sphere, whatever its size
        back = 0.5 * np.linalg.norm(self.object_size) + self.get_parameter('approach_margin').value
        pre, lift = grasp.copy(), grasp.copy()
        pre[:3, 3] -= back * grasp[:3, 0]
        lift[:3, 3] += [0.0, 0.0, self.get_parameter('lift_height').value]
        return {'pre': pre, 'grasp': grasp, 'lift': lift}

    def reject(self, reason):
        self.rejections[reason] += 1
        self.get_logger().warn(f'Rejected: {reason}')
        return None

    def pre_solutions(self, target):
        """Distinct collision-free IK solutions for `target`, nearest the current joints first."""
        start = np.array(self.joint_positions)
        rng = np.random.default_rng(0)
        seeds = [start] + [rng.uniform(-np.pi, np.pi, len(ARM_JOINTS))
                           for _ in range(self.get_parameter('ik_seeds').value)]
        found = []
        for seed in seeds:
            joints = self.solve_ik(target, seed)
            if joints is None:
                continue
            joints = np.array(joints)
            if all(np.abs(joints - other).max() > 0.05 for other in found):
                found.append(joints)
        found.sort(key=lambda q: np.abs(q - start)[:5].max())
        return found

    def check_branch(self, joints, goals):
        """Checks the whole pick for one pre-grasp solution. Returns (plan, reason)."""
        start = list(self.joint_positions)
        up_joints = self.get_parameter('up_joints').value

        self.set_contact_allowed(False)
        approach = self.plan_line(joints, goals['grasp'])
        if approach is None:
            return None, 'straight approach is blocked'

        self.set_contact_allowed(True)  # once closed, the fingers touch the object by design
        lift = self.plan_line(approach[1], goals['lift'])
        if lift is None:
            return None, 'straight lift is blocked'
        if self.plan_to_joints(lift[1], up_joints) is None:
            return None, 'cannot reach the up pose from the lift pose'

        # The planner call is the slow one, so it runs last
        self.set_contact_allowed(False)
        pre_trajectory = self.plan_to_joints(start, joints)
        if pre_trajectory is None:
            return None, 'no collision-free path to the pre-grasp'
        return {'goals': goals, 'pre_trajectory': pre_trajectory}, None

    def screen(self, grasp):
        """Checks one hand pose end to end without moving. Returns a plan dict or None."""
        self.set_contact_allowed(False)
        found = self.find_envelope(grasp)
        if found is None:
            return self.reject('no contact-free hand position around the object')
        grasp, width, retreat = found
        centre = grasp[:3, 3] + grasp[:3, :3] @ self.hand_centre
        base = self.object_centre[2] - self.object_size[2] / 2.0
        self.get_logger().info(
            f'Closing centre {centre[2] - base:.3f} m above the object base '
            f'(object height {self.object_size[2]:.3f} m)')
        self.get_logger().info(
            f'Approach z {grasp[2, 0]:+.2f}, closing axis z {(grasp[:3, :3] @ self.hand_axis)[2]:+.2f}')
        self.get_logger().info(
            f'Hand envelopes the object: {width:.3f} m free across, backed off {retreat:.3f} m')

        goals = self.targets(grasp)
        solutions = self.pre_solutions(goals['pre'])
        if not solutions:
            return self.reject('no collision-free IK solution for the pre-grasp')
        for joints in solutions:
            plan, reason = self.check_branch(joints, goals)
            if plan is not None:
                self.get_logger().info(
                    f'Hand pose accepted ({len(solutions)} pre-grasp solutions found)')
                return plan
            self.reject(reason)
        return None

    def execute_pick(self, plan):
        goals = plan['goals']

        self.set_contact_allowed(False)
        self.get_logger().info('Pre-grasp')
        if not self.execute_trajectory(plan['pre_trajectory']):
            raise RuntimeError('Pre-grasp failed')
        self.get_logger().info('Approach')
        self.line_move(goals['grasp'], 'Approach')

        self.set_contact_allowed(True)
        self.get_logger().info('Close gripper')
        if not self.close_until_contact():
            raise RuntimeError('Close gripper failed')

        for frame in (*self.get_parameter('hand_pair_frames').value,
                      self.get_parameter('hand_opposing_frame').value):
            p = lookup_matrix(self.tf_buffer, self.frame_id, frame)[:3, 3] - self.object_centre
            self.get_logger().info(f'{frame} vs object centre: {np.round(p, 3).tolist()}')

        self.get_logger().info('Lift')
        self.line_move(goals['lift'], 'Lift')

        self.get_logger().info('Up')
        self.spin_for(0.3)
        if not self.move_to_joints(self.get_parameter('up_joints').value):
            raise RuntimeError('Up move failed')

    # ---------- sequence ----------

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

    def run(self):
        self.get_logger().info('Waiting for candidates, cloud, /joint_states and TF...')
        while rclpy.ok() and not self.ready():
            rclpy.spin_once(self, timeout_sec=0.1)

        self.frame_id = self.candidates.header.frame_id
        try:
            link_from_tcp = lookup_matrix(
                self.tf_buffer, self.get_parameter('ik_link').value,
                self.get_parameter('tcp_frame').value)
        except TransformException as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return False
        self.tcp_to_link = np.linalg.inv(link_from_tcp)

        # Open first: the hand geometry below must be measured with the fingers open
        if not self.move_gripper(self.get_parameter('gripper_open').value):
            return False
        self.spin_for(0.5)

        try:
            self.hand_geometry()
            self.gripper_links = self.discover_gripper_links()
            self.remove_object()
            self.add_object()
            self.add_support()
        except (TransformException, RuntimeError) as e:
            self.get_logger().error(str(e))
            return False

        proposals = [pose_to_matrix(p) for p in self.candidates.poses]
        proposals.append(self.top_down_grasp())
        tried = []
        for rank, proposal in enumerate(proposals, 1):
            for grasp in self.hand_poses(proposal):
                if any(is_similar(grasp, other) for other in tried):
                    continue
                tried.append(grasp)
                label = 'top-down fallback' if rank == len(proposals) else f'{rank}/{len(proposals)}'
                self.get_logger().info(f'Proposal {label}')
                try:
                    plan = self.screen(grasp)
                    if plan is None:
                        continue
                    self.execute_pick(plan)
                except RuntimeError as e:
                    self.get_logger().error(str(e))
                    return False
                self.get_logger().info('Pick complete')
                return True

        summary = ', '.join(f'{n}x {reason}' for reason, n in self.rejections.most_common())
        self.get_logger().error(f'No hand pose gave a feasible pick ({summary})')
        return False


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