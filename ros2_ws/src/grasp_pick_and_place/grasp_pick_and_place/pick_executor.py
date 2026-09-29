import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from collections import Counter
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry, AllowedCollisionMatrix, AttachedCollisionObject,
    CollisionObject, Constraints, JointConstraint, PlanningScene,
    PlanningSceneComponents, RobotState)
from moveit_msgs.srv import (
    ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetPositionIK, GetStateValidity)
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from shape_msgs.msg import SolidPrimitive
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
OBJECT_ID = 'target_object'


class PickExecutor(Node):
    """Picks one object: open, pre-grasp, straight approach, close, straight lift, up.

    Each candidate grasp is checked end to end (IK, pre-grasp plan, straight-line approach,
    lift and up) before anything moves. The object is a collision object in MoveIt's scene;
    gripper-object contact is only allowed for the last `contact_distance` of the approach
    and after that the object is attached to the gripper.
    """

    def __init__(self):
        super().__init__('pick_executor')

        self.declare_parameter('group_name', 'ur_manipulator')
        self.declare_parameter('ik_link', 'tool0')
        self.declare_parameter('tcp_frame', 'gripper_tool0')
        self.declare_parameter('clearance_step', 0.02)
        self.declare_parameter('max_clearance', 0.30)
        self.declare_parameter('clearance_margin', 0.03)
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
        self.declare_parameter('model_object', True)
        self.declare_parameter('object_radius', 0.10)
        self.declare_parameter('object_margin', 0.01)
        self.declare_parameter('gripper_link_prefix', 'gripper_')
        self.declare_parameter('gripper_links', '')  # comma-separated override
        self.declare_parameter('gripper_open', [0.05, 0.05, 0.05])
        self.declare_parameter('gripper_duration', 1.5)
        self.declare_parameter('close_step', 0.03)
        self.declare_parameter('close_period', 0.1)
        self.declare_parameter('close_limit', [1.2, 1.2, 1.2])
        self.declare_parameter('contact_lag', 0.08)
        self.declare_parameter('squeeze_margin', 0.02)
        self.declare_parameter('close_timeout', 15.0)
        self.declare_parameter('motion_eps', 0.01)
        self.declare_parameter('stall_count', 2)

        self.model_object = self.get_parameter('model_object').value
        self.candidates = None
        self.scene_points = None
        self.scene_frame = None
        self.joint_positions = None
        self.finger_positions = None
        self.joint_map = {}
        self.gripper_links = []
        self.rejections = Counter()

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

    def object_box(self, center):
        """Axis-aligned box around the cloud points near `center`, or None."""
        radius = self.get_parameter('object_radius').value
        margin = self.get_parameter('object_margin').value
        near = np.linalg.norm(self.scene_points[:, :2] - center[:2], axis=1) < radius
        points = self.scene_points[near]
        if points.shape[0] < 20:
            return None
        low = points.min(axis=0) - margin
        high = points.max(axis=0) + margin
        low[2] = points[:, 2].min()  # keep the bottom on the table
        return low, high

    def add_object(self, low, high):
        obj = CollisionObject()
        obj.header.frame_id = self.scene_frame
        obj.id = OBJECT_ID
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(v) for v in (high - low)]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, (low + high) / 2.0)
        pose.orientation.w = 1.0
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        self.apply_scene(scene)

    def remove_object(self):
        """Clears a target_object left over from an earlier run (world or attached)."""
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        current = self.call_service(self.get_scene_client, request).scene

        scene = PlanningScene()
        scene.is_diff = True

        if any(a.object.id == OBJECT_ID
               for a in current.robot_state.attached_collision_objects):
            detach = AttachedCollisionObject()
            detach.link_name = self.get_parameter('ik_link').value
            detach.object.id = OBJECT_ID
            detach.object.operation = CollisionObject.REMOVE
            scene.robot_state.is_diff = True
            scene.robot_state.attached_collision_objects = [detach]

        if any(o.id == OBJECT_ID for o in current.world.collision_objects):
            remove = CollisionObject()
            remove.id = OBJECT_ID
            remove.operation = CollisionObject.REMOVE
            scene.world.collision_objects = [remove]

        if scene.robot_state.attached_collision_objects or scene.world.collision_objects:
            self.apply_scene(scene)

    def set_contact_allowed(self, allowed):
        """Allow or forbid gripper-object contact in MoveIt's collision matrix."""
        if not self.model_object:
            return
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

    def attach_object(self):
        if not self.model_object:
            return
        attached = AttachedCollisionObject()
        attached.link_name = self.get_parameter('ik_link').value
        attached.object.id = OBJECT_ID
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = list(self.gripper_links) + [attached.link_name]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        self.apply_scene(scene)

    # ---------- IK and motion ----------

    def solve_ik(self, pose_matrix, frame_id, seed, avoid_collisions=True):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self.get_parameter('group_name').value
        ik.ik_link_name = self.get_parameter('ik_link').value
        ik.avoid_collisions = avoid_collisions
        timeout = self.get_parameter('ik_timeout').value
        ik.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1) * 1e9))
        ik.robot_state.joint_state = JointState(
            name=ARM_JOINTS, position=[float(p) for p in seed])
        ik.pose_stamped.header.frame_id = frame_id
        ik.pose_stamped.pose = matrix_to_pose(pose_matrix)

        response = self.call_service(self.ik_client, request)
        if response.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().debug(f'IK failed (code {response.error_code.val})')
            return None
        solved = dict(zip(response.solution.joint_state.name,
                          response.solution.joint_state.position))
        joints = [solved[n] for n in ARM_JOINTS]
        # IK may return an angle a full turn away from the seed; use the equivalent one nearest it
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

    def plan_line(self, start_joints, target, frame_id, tcp_to_link):
        """Straight-line tool path to `target`. Returns (trajectory, end_joints) or None."""
        request = GetCartesianPath.Request()
        request.header.frame_id = frame_id
        request.group_name = self.get_parameter('group_name').value
        request.link_name = self.get_parameter('ik_link').value
        request.start_state = RobotState()
        request.start_state.joint_state = JointState(
            name=ARM_JOINTS, position=[float(p) for p in start_joints])
        request.waypoints = [matrix_to_pose(target @ tcp_to_link)]
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

        self.get_logger().warn(
            f'Straight-line path only {fraction:.0%} feasible '
            f'(code {response.error_code.val})')
        
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

    def line_move(self, target, frame_id, tcp_to_link, label):
        """Straight-line move from the actual current state; raises if it fails."""
        self.spin_for(0.3)  # let /joint_states settle before planning from the current state
        plan = self.plan_line(self.joint_positions, target, frame_id, tcp_to_link)
        if plan is None or not self.execute_trajectory(plan[0]):
            raise RuntimeError(f'{label} failed')

    # ---------- gripper ----------

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

    def send_finger_targets(self, positions, duration):
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINTS
        goal.trajectory.points = [point]
        self.gripper_client.send_goal_async(goal)

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
                blocked = (
                    has_moved and still[i] >= needed and target[i] - measured > lag)
                if blocked:
                    stalled[i] = True
                    target[i] = measured + margin
                    self.get_logger().info(f'{name} stalled at {measured:.3f}')
                elif target[i] >= limits[i]:
                    self.get_logger().error(f'{name} reached its limit without contact')
                    return False
                else:
                    target[i] = min(target[i] + step, limits[i])

            self.get_logger().info(
                f'target={[round(t, 3) for t in target]} '
                f'measured={[round(m, 3) for m in self.finger_positions]}')

            self.send_finger_targets(target, period)
            self.spin_for(period)
            if all(stalled):
                return True
        self.get_logger().error('Gripper close timed out')
        return False

    # ---------- one candidate: check, then execute ----------

    def targets(self, grasp, back):
        approach = grasp[:3, 0]
        margin = self.get_parameter('clearance_margin').value

        def shifted(offset):
            m = grasp.copy()
            m[:3, 3] += offset
            return m

        return {
            'pre': shifted(-(back + margin) * approach),
            'standoff': shifted(-back * approach),
            'grasp': grasp,
            'lift': shifted(np.array([0.0, 0.0, self.get_parameter('lift_height').value])),
        }

    def reject(self, reason):
        self.rejections[reason] += 1
        self.get_logger().warn(f'Rejected: {reason}')
        return None

    def pre_solutions(self, target, frame_id, tcp_to_link):
        """Distinct IK solutions for `target`, nearest to the current joints first."""
        start = np.array(self.joint_positions)
        rng = np.random.default_rng(0)
        seeds = [start] + [rng.uniform(-np.pi, np.pi, len(ARM_JOINTS))
                           for _ in range(self.get_parameter('ik_seeds').value)]
        found = []
        for seed in seeds:
            joints = self.solve_ik(target @ tcp_to_link, frame_id, seed)
            if joints is None:
                continue
            joints = np.array(joints)
            if all(np.abs(joints - other).max() > 0.05 for other in found):
                found.append(joints)
        found.sort(key=lambda q: np.abs(q - start)[:5].max())
        return found

    def check_branch(self, joints, goals, frame_id, tcp_to_link):
        """Checks the whole pick for one pre-grasp solution. Returns (plan, reason)."""
        start = list(self.joint_positions)
        up_joints = self.get_parameter('up_joints').value

        self.set_contact_allowed(False)
        approach = self.plan_line(joints, goals['standoff'], frame_id, tcp_to_link)
        if approach is None:
            return None, 'straight approach is blocked'

        self.set_contact_allowed(True)
        final = self.plan_line(approach[1], goals['grasp'], frame_id, tcp_to_link)
        if final is None:
            return None, 'final approach is blocked'
        lift = self.plan_line(final[1], goals['lift'], frame_id, tcp_to_link)
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

    def screen(self, pose, frame_id, tcp_to_link):
        """Checks one candidate end to end without moving. Returns a plan dict or None."""
        grasp = pose_to_matrix(pose)
        back, reason = self.clearance(grasp, frame_id, tcp_to_link)
        if back is None:
            return self.reject(reason)
        self.get_logger().info(f'Open hand clears the object {back:.2f} m back')
        goals = self.targets(grasp, back)
        self.set_contact_allowed(False)
        solutions = self.pre_solutions(goals['pre'], frame_id, tcp_to_link)
        if not solutions:
            return self.reject('no collision-free IK solution for the pre-grasp')
        for joints in solutions:
            plan, reason = self.check_branch(joints, goals, frame_id, tcp_to_link)
            if plan is not None:
                self.get_logger().info(
                    f'Candidate accepted ({len(solutions)} pre-grasp solutions found)')
                return plan
            self.reject(reason)
        return None

    def execute_pick(self, plan, frame_id, tcp_to_link):
        goals = plan['goals']

        self.set_contact_allowed(False)
        self.get_logger().info('Pre-grasp')
        if not self.execute_trajectory(plan['pre_trajectory']):
            raise RuntimeError('Pre-grasp failed')
        self.get_logger().info('Approach')
        self.line_move(goals['standoff'], frame_id, tcp_to_link, 'Approach')

        self.set_contact_allowed(True)
        self.get_logger().info('Final approach')
        self.line_move(goals['grasp'], frame_id, tcp_to_link, 'Final approach')

        self.get_logger().info('Close gripper')
        if not self.close_until_contact():
            raise RuntimeError('Gripper close failed')
        self.attach_object()

        self.get_logger().info('Lift')
        self.line_move(goals['lift'], frame_id, tcp_to_link, 'Lift')

        self.get_logger().info('Up')
        self.spin_for(0.3)
        if not self.move_to_joints(self.get_parameter('up_joints').value):
            raise RuntimeError('Up move failed')

    def order_by_reach(self, poses, frame_id, tcp_to_link):
        """Sorts candidates by the joint swing their pre-grasp needs, cheapest first."""
        start = list(self.joint_positions)
        self.set_contact_allowed(True)
        ranked = []
        for pose in poses:
            pre = self.targets(pose_to_matrix(pose))['pre']
            joints = self.solve_ik(pre @ tcp_to_link, frame_id, start)
            cost = (float('inf') if joints is None
                    else max(abs(a - b) for a, b in zip(joints[:5], start[:5])))
            ranked.append((cost, pose))
        ranked.sort(key=lambda item: item[0])  # stable: ties keep GraspNet's score order
        return [pose for _, pose in ranked]

    def touches_object(self, joints):
        """True if the robot in this arm configuration touches the object (current gripper state)."""
        positions = dict(self.joint_map)
        positions.update(zip(ARM_JOINTS, (float(q) for q in joints)))
        request = GetStateValidity.Request()
        request.group_name = self.get_parameter('group_name').value
        request.robot_state.joint_state = JointState(
            name=list(positions), position=list(positions.values()))
        response = self.call_service(self.validity_client, request)
        return any(OBJECT_ID in (c.contact_body_1, c.contact_body_2)
                   for c in response.contacts)

    def clearance(self, grasp, frame_id, tcp_to_link):
        """Distance to back off along the approach until the open hand is clear of the object.

        Returns (distance, None) or (None, reason)."""
        step = self.get_parameter('clearance_step').value
        if not self.model_object:
            return step, None
        limit = self.get_parameter('max_clearance').value
        approach = grasp[:3, 0]
        self.set_contact_allowed(False)
        seed = list(self.joint_positions)
        back = step
        while back <= limit + 1e-9:
            target = grasp.copy()
            target[:3, 3] -= back * approach
            joints = self.solve_ik(
                target @ tcp_to_link, frame_id, seed, avoid_collisions=False)
            if joints is None:
                return None, 'pose on the way out of the grasp is unreachable'
            if not self.touches_object(joints):
                return back, None
            seed = joints
            back += step
        return None, 'open hand still touches the object at the maximum back-off'

    # ---------- sequence ----------

    def ready(self):
        return (
            self.candidates is not None and self.candidates.poses
            and (self.scene_points is not None or not self.model_object)
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

        frame_id = self.candidates.header.frame_id
        try:
            link_from_tcp = lookup_matrix(
                self.tf_buffer, self.get_parameter('ik_link').value,
                self.get_parameter('tcp_frame').value)
        except TransformException as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return False
        tcp_to_link = np.linalg.inv(link_from_tcp)

        if self.model_object:
            try:
                self.gripper_links = self.discover_gripper_links()
                self.remove_object()
                best = self.candidates.poses[0].position
                box = self.object_box(np.array([best.x, best.y, best.z]))
                if box is None:
                    self.get_logger().error('Too few cloud points near the best grasp')
                    return False
                self.add_object(*box)
            except RuntimeError as e:
                self.get_logger().error(str(e))
                return False
            low, high = box
            self.get_logger().info(
                f'Object box size {np.round(high - low, 3).tolist()} '
                f'centre {np.round((low + high) / 2, 3).tolist()}; '
                f'{len(self.gripper_links)} gripper links')

        if not self.move_gripper(self.get_parameter('gripper_open').value):
            return False

        total = len(self.candidates.poses)
        for rank, pose in enumerate(self.candidates.poses, 1):
            self.get_logger().info(f'Candidate {rank}/{total}')
            try:
                plan = self.screen(pose, frame_id, tcp_to_link)
                if plan is None:
                    continue
                self.execute_pick(plan, frame_id, tcp_to_link)
            except RuntimeError as e:
                self.get_logger().error(str(e))
                return False
            self.get_logger().info('Pick complete')
            return True
        summary = ', '.join(f'{n}x {reason}' for reason, n in self.rejections.most_common())
        self.get_logger().error(f'No candidate gave a feasible pick ({summary})')


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