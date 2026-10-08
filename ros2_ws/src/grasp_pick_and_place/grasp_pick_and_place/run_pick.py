import os
import signal
import subprocess
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from tf2_ros import TransformException

import numpy as np

from grasp_pick_and_place.geometry import is_similar
from grasp_pick_and_place.pick_executor import PickExecutor
from grasp_pick_and_place.transform_utils import lookup_matrix, pose_to_matrix

PKG = 'grasp_pick_and_place'
BACKGROUND_NODES = ['scene_snapshot', 'grasp_proposer']
CAPTURE_SRV = '/capture_scene'
POSE_TOPIC = '/grasp_pose'
READY_TIMEOUT = 120.0
POSE_TIMEOUT = 20.0  # per capture attempt
CAPTURE_ATTEMPTS = 3


# ---------- process helpers ----------

def start(executable):
    return subprocess.Popen(['ros2', 'run', PKG, executable], start_new_session=True)


def stop(proc):
    if proc.poll() is not None:
        return
    group = os.getpgid(proc.pid)
    os.killpg(group, signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(group, signal.SIGKILL)


def spin_until(node, condition, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        rclpy.spin_once(node, timeout_sec=0.1)
    return condition()


# ---------- sequence steps ----------

def capture_scene(robot):
    """Triggers scene_snapshot until grasp_proposer publishes a pose."""
    client = robot.create_client(Trigger, CAPTURE_SRV)
    if not spin_until(robot, client.service_is_ready, READY_TIMEOUT):
        raise RuntimeError(f'{CAPTURE_SRV} not available')
    if not spin_until(robot, lambda: robot.count_publishers(POSE_TOPIC) > 0, READY_TIMEOUT):
        raise RuntimeError(f'{POSE_TOPIC} has no publisher')

    received = []
    robot.create_subscription(PoseStamped, POSE_TOPIC, received.append, 10)
    spin_until(robot, lambda: False, 2.0)  # let the subscription connect

    for attempt in range(1, CAPTURE_ATTEMPTS + 1):
        robot.get_logger().info(f'Capturing scene (attempt {attempt}/{CAPTURE_ATTEMPTS})')
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(robot, future)
        if spin_until(robot, lambda: bool(received), POSE_TIMEOUT):
            break
    else:
        raise RuntimeError('No grasp pose received')

    pose = received[0]
    robot.get_logger().info(
        f'Best grasp in {pose.header.frame_id}: '
        f'position=({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, '
        f'{pose.pose.position.z:.3f}) '
        f'orientation=({pose.pose.orientation.x:.3f}, {pose.pose.orientation.y:.3f}, '
        f'{pose.pose.orientation.z:.3f}, {pose.pose.orientation.w:.3f})')


def prepare(robot):
    """Waits for inputs, opens the gripper, measures the hand, models object and table."""
    robot.get_logger().info('Waiting for candidates, cloud, /joint_states and TF...')
    while rclpy.ok() and not robot.ready():
        rclpy.spin_once(robot, timeout_sec=0.1)

    robot.frame_id = robot.candidates.header.frame_id
    link_from_tcp = lookup_matrix(
        robot.tf_buffer, robot.get_parameter('ik_link').value,
        robot.get_parameter('tcp_frame').value)
    robot.tcp_to_link = np.linalg.inv(link_from_tcp)

    # Open first: the hand geometry must be measured with the fingers open
    if not robot.move_gripper(robot.get_parameter('gripper_open').value):
        raise RuntimeError('Could not open the gripper')
    robot.spin_for(0.5)

    robot.hand_geometry()
    robot.gripper_links = robot.discover_gripper_links()
    robot.remove_object()
    robot.add_object()
    robot.add_support()


def place_all(robot):
    """Places the open hand for every proposal without moving, ranked by closing-centre offset.
    Returns (placed, proposal_count)."""
    proposals = [pose_to_matrix(p) for p in robot.candidates.poses]
    proposals.append(robot.top_down_grasp())

    placed, tried = [], []
    for rank, proposal in enumerate(proposals, 1):
        for grasp in robot.hand_poses(proposal):
            if any(is_similar(grasp, other) for other in tried):
                continue
            tried.append(grasp)
            result = robot.place_hand(grasp)
            if result is not None:
                placed.append((result[3], rank, *result[:3]))
    placed.sort(key=lambda item: item[0])  # stable: GraspNet order breaks ties
    robot.get_logger().info(f'{len(placed)} of {len(tried)} hand poses fit around the object')
    return placed, len(proposals)


def execute_pick(robot, plan):
    goals = plan['goals']

    robot.set_contact_allowed(False)
    robot.get_logger().info('Pre-grasp')
    if not robot.execute_trajectory(plan['pre_trajectory']):
        raise RuntimeError('Pre-grasp failed')
    robot.get_logger().info('Approach')
    robot.line_move(goals['grasp'], 'Approach')

    robot.set_contact_allowed(True)
    robot.get_logger().info('Close gripper')
    if not robot.close_until_contact():
        raise RuntimeError('Close gripper failed')

    robot.get_logger().info('Lift')
    robot.line_move(goals['lift'], 'Lift')

    robot.get_logger().info('Up')
    robot.spin_for(0.3)
    up = robot.plan_joint_path(
        list(robot.joint_positions), robot.get_parameter('up_joints').value)
    if up is None or not robot.execute_trajectory(up):
        raise RuntimeError('Up move failed')


def pick(robot, placed, proposal_count):
    """Best placement first; executes the first one whose whole pick is feasible."""
    for offset, rank, grasp, width, retreat in placed:
        label = 'top-down fallback' if rank == proposal_count else f'{rank}/{proposal_count}'
        robot.get_logger().info(
            f'Proposal {label}: closing centre {offset:.2f} of the half-diagonal from the '
            f'centroid, {width:.3f} m free across, backed off {retreat:.3f} m')
        plan = robot.plan_pick(grasp)
        if plan is None:
            continue
        execute_pick(robot, plan)
        robot.get_logger().info('Pick complete')
        return

    summary = ', '.join(f'{n}x {reason}' for reason, n in robot.rejections.most_common())
    raise RuntimeError(f'No hand pose gave a feasible pick ({summary})')


# ---------- entry point ----------

def main():
    rclpy.init()
    background = []
    robot = None
    try:
        robot_log = rclpy.logging.get_logger('run_pick')
        robot_log.info('Starting ' + ' and '.join(BACKGROUND_NODES))
        background = [start(name) for name in BACKGROUND_NODES]

        robot = PickExecutor()
        capture_scene(robot)
        prepare(robot)
        placed, proposal_count = place_all(robot)
        pick(robot, placed, proposal_count)
    except (RuntimeError, TransformException) as err:
        rclpy.logging.get_logger('run_pick').error(str(err))
    except KeyboardInterrupt:
        pass
    finally:
        for proc in background:
            stop(proc)
        if robot is not None:
            robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()