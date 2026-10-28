import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from grasp_pick_and_place.config import GRIPPER_JOINTS, HARDWARE_MODE
from grasp_pick_and_place.transform_utils import lookup_matrix


class GripperMixin:
    """Finger commands, close-until-contact, hand geometry from TF."""

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
        return True

    def close_gripper(self):
        step = self.get_parameter('close_step').value
        period = self.get_parameter('close_period').value
        limits = list(self.get_parameter('close_limit').value)
        lag = self.get_parameter('contact_lag').value
        margin = self.get_parameter(
            'squeeze_margin_sphere' if self.sphere else 'squeeze_margin').value
        eps = self.get_parameter('motion_eps').value
        needed = self.get_parameter('stall_count').value
        deadline = time.monotonic() + self.get_parameter('close_timeout').value
        t0 = time.monotonic()

        start = list(self.finger_positions)
        target = list(start)
        previous = list(start)
        still = [0] * len(GRIPPER_JOINTS)
        stalled = [False] * len(GRIPPER_JOINTS)
        moved = [False] * len(GRIPPER_JOINTS)
        while rclpy.ok() and time.monotonic() < deadline:
            for i, name in enumerate(GRIPPER_JOINTS):
                if stalled[i]:
                    continue
                measured = self.finger_positions[i]
                still[i] = still[i] + 1 if abs(measured - previous[i]) < eps else 0
                previous[i] = measured
                # Latched: a finger pushed back to its open stop has still moved earlier
                moved[i] = moved[i] or measured - start[i] > step
                blocked = moved[i] and still[i] >= needed and target[i] - measured > lag
                if blocked:
                    self.get_logger().info(
                        f'{name} stalled at {measured:.3f} (travel {measured - start[i]:.3f})')
                    stalled[i] = True
                    target[i] = measured + margin
                elif target[i] >= limits[i]:
                    self.get_logger().error(f'{name} reached its limit without contact')
                    return False
                else:
                    target[i] = min(target[i] + step, limits[i])

            self.get_logger().info(   # add, before send_goal_async
                f'close t={time.monotonic() - t0:.2f} meas={np.round(self.finger_positions, 3).tolist()} '
                f'tgt={np.round(target, 3).tolist()} still={still}')

            self.gripper_client.send_goal_async(self.finger_goal(target, period))
            self.spin_for(period)
            if all(stalled):
                return True
        self.get_logger().error('Gripper close timed out')
        return False

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
        direction[0] = 0.0  # x is the approach axis of the grasp frame
        centre[0] = self.get_parameter('hand_centre_depth').value
        if HARDWARE_MODE == 'real':
            centre[1] += 0.0385  # temporary test: restore pre-URDF closing centre
        gap = float(np.linalg.norm(direction))
        self.hand_gap = gap
        self.get_logger().info(f'hand gap {gap:.3f}')
        self.hand_axis = direction / gap
        self.hand_centre = centre

    def ensure_open(self):
        return self.move_gripper(self.get_parameter('gripper_open').value)

    def ensure_closed(self):
        return self.move_gripper(self.get_parameter('gripper_closed').value)