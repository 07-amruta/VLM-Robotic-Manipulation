import numpy as np
from builtin_interfaces.msg import Duration
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath, GetPositionIK, GetStateValidity
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from grasp_pick_and_place.config import ARM_JOINTS, GRIPPER_JOINTS, MOVEIT_SUCCESS
from grasp_pick_and_place.transform_utils import matrix_to_pose


class MotionMixin:
    """IK, validity checks, joint-space and Cartesian planning, trajectory execution."""

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

    def plan_to_joints(self, start_joints, goal_joints):
        """Plan only (OMPL), from a given start state. Returns a RobotTrajectory or None."""
        goal = self.joint_goal(goal_joints)
        goal.request.start_state.joint_state = JointState(
            name=ARM_JOINTS, position=[float(p) for p in start_joints])
        goal.planning_options.plan_only = True
        result = self.run_goal(self.move_group_client, goal)
        if result is None or result.error_code.val != MOVEIT_SUCCESS:
            return None
        return result.planned_trajectory

    def validity_at(self, joints, fingers=None):
        """MoveIt state-validity response for an arm configuration (and optional finger values)."""
        positions = dict(self.joint_map)
        positions.update(zip(ARM_JOINTS, (float(q) for q in joints)))
        if fingers is not None:
            positions.update(zip(GRIPPER_JOINTS, (float(q) for q in fingers)))
        request = GetStateValidity.Request()
        request.group_name = self.get_parameter('group_name').value
        request.robot_state.joint_state = JointState(
            name=list(positions), position=list(positions.values()))
        return self.call_service(self.validity_client, request)

    def plan_joint_line(self, start_joints, goal_joints):
        """Straight line in joint space, every state collision-checked. RobotTrajectory or None."""
        start, goal = np.array(start_joints, dtype=float), np.array(goal_joints, dtype=float)
        step = self.get_parameter('joint_line_step').value
        n = max(int(np.ceil(np.abs(goal - start).max() / step)) + 1, 2)
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ARM_JOINTS
        for s in np.linspace(0.0, 1.0, n):
            q = start + s * (goal - start)
            if not self.validity_at(q).valid:
                return None
            trajectory.joint_trajectory.points.append(
                JointTrajectoryPoint(positions=[float(v) for v in q]))
        self.time_parameterize(trajectory, self.get_parameter('joint_line_speed').value)
        return trajectory

    def plan_joint_path(self, start_joints, goal_joints):
        """Straight joint-space line if collision-free, else OMPL. RobotTrajectory or None."""
        trajectory = self.plan_joint_line(start_joints, goal_joints)
        if trajectory is None:
            trajectory = self.plan_to_joints(start_joints, goal_joints)
        return trajectory

    def time_parameterize(self, trajectory, speed=None):
        """Positions-only timing: constant joint speed, strictly increasing stamps."""
        speed = speed or self.get_parameter('cartesian_joint_speed').value
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