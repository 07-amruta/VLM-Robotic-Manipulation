import os
import signal
import subprocess
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

import numpy as np

from grasp_pick_and_place.config import (
    CAMERA, HARDWARE_MODE, MANUAL_SCENE, SAFETY, SCENE, SCENE_SOURCE, TARGET, ARM_JOINTS)
from grasp_pick_and_place.geometry import is_similar
from grasp_pick_and_place.pick_executor import PickExecutor
from grasp_pick_and_place.transform_utils import lookup_matrix, pose_to_matrix

PKG = 'grasp_pick_and_place'
CAPTURE_SRV = '/capture_scene'
POSE_TOPIC = '/grasp_pose'
READY_TIMEOUT = 120.0
POSE_TIMEOUT = 20.0  # per capture attempt
CAPTURE_ATTEMPTS = 3
CAPTURE_WAIT = 30.0          # s, retry a failed capture (perception warm-up) before giving up
CAPTURE_RETRY_PERIOD = 1.0   # s


def background_nodes():
    """Nodes started with the sequence: the scene source."""
    if SCENE_SOURCE == 'manual':
        return {'manual_scene': {**MANUAL_SCENE, 'support_top': SCENE['support_top']}}
    target = {k: v for k, v in TARGET.items() if v != ''}
    return {'scene_snapshot': {**CAMERA, 'objects_topic': '/scene_objects',
                               'obstacles_topic': '/scene_obstacle_clusters',
                               'planes_topic': '/support_planes', **target},
            'grasp_proposer': {'optical_frame': CAMERA['optical_frame']}}


# ---------- process helpers ----------

def start(executable, params):
    command = ['ros2', 'run', PKG, executable]
    if params:
        command += ['--ros-args']
        for name, value in params.items():
            command += ['-p', f'{name}:={value}']
    return subprocess.Popen(command, start_new_session=True)


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
def trigger_capture(robot, client):
    """Calls the capture service until it succeeds or CAPTURE_WAIT elapses."""
    deadline = time.monotonic() + CAPTURE_WAIT
    while True:
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(robot, future)
        result = future.result()
        robot.get_logger().info(f'capture: {result.message}')
        if result.success or time.monotonic() >= deadline:
            return result.success
        spin_until(robot, lambda: False, CAPTURE_RETRY_PERIOD)

def capture_scene(robot):
    """Triggers the scene source until a grasp pose is published."""
    client = robot.create_client(Trigger, CAPTURE_SRV)
    if not spin_until(robot, client.service_is_ready, READY_TIMEOUT):
        raise RuntimeError(f'{CAPTURE_SRV} not available')
    if not spin_until(robot, lambda: robot.count_publishers(POSE_TOPIC) > 0, READY_TIMEOUT):
        raise RuntimeError(f'{POSE_TOPIC} has no publisher')

    received = []
    robot.create_subscription(PoseStamped, POSE_TOPIC, received.append, 10)
    target = os.environ.get('TARGET_CLASS', '').strip()
    if target:  # unset = request sent by hand / another node
        request_pub = robot.create_publisher(
            String, '/target_request',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        request_pub.publish(String(data=target))
    spin_until(robot, lambda: False, 2.0)  # let the subscription connect

    for attempt in range(1, CAPTURE_ATTEMPTS + 1):
        robot.get_logger().info(f'Capturing scene (attempt {attempt}/{CAPTURE_ATTEMPTS})')
        if trigger_capture(robot, client) and spin_until(
                robot, lambda: bool(received), POSE_TIMEOUT):
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
    if not robot.ensure_open():
        raise RuntimeError('Could not open the gripper')
    robot.spin_for(0.5)

    robot.hand_geometry()
    robot.gripper_links = robot.discover_gripper_links()
    robot.remove_object()
    robot.remove_obstacles()
    robot.remove_support()
    robot.add_object()
    robot.add_obstacles()
    robot.add_support()
    robot.add_floor()


def place_all(robot):
    """Places the open hand for every proposal without moving, ranked by closing-centre offset.
    Returns (placed, proposal_count)."""
    proposals = [pose_to_matrix(p) for p in robot.candidates.poses]
    proposals.append(robot.top_down_grasp())

    placed, tried = [], []
    log = robot.get_logger()
    for rank, proposal in enumerate(proposals, 1):
        label = 'fallback' if rank == len(proposals) else str(rank)
        poses = robot.hand_poses(proposal)
        if not poses:
            log.info(f'[{label}] no hand pose')
        for grasp in poses:
            az = grasp[2, 0]  # approach z; -1 = straight down
            if any(is_similar(grasp, other) for other in tried):
                log.info(f'[{label}] duplicate az={az:.2f}')
                continue
            tried.append(grasp)
            result = robot.place_hand(grasp)
            if result is None:
                log.info(f'[{label}] rejected az={az:.2f}')
                continue
            log.info(f'[{label}] placed offset={result[3]:.2f} az={az:.2f}')
            placed.append((result[3], rank, *result[:3]))
            
    placed.sort(key=lambda item: item[0])  # stable: GraspNet order breaks ties
    robot.get_logger().info(f'{len(placed)} of {len(tried)} hand poses fit around the object')
    return placed, len(proposals)


def confirm_stage(label):
    """Operator gate before one stage of the pick (real hardware)."""
    if SAFETY['confirm_each_stage'] and input(
            f'{label}: y to run, anything else aborts ').strip().lower() != 'y':
        raise RuntimeError(f'{label} declined')

def log_fingers(robot, label):
    robot.spin_for(0.3)
    robot.get_logger().info(f'TEMP fingers {label} {np.round(robot.finger_positions, 3).tolist()}')

def execute_pick(robot, plan):
    goals = plan['goals']

    robot.set_contact_allowed(False)
    confirm_stage('Pre-grasp')
    robot.get_logger().info('Pre-grasp')
    if not robot.execute_trajectory(plan['pre_trajectory']):
        raise RuntimeError('Pre-grasp failed')
    confirm_stage('Approach')
    robot.get_logger().info('Approach')
    robot.spin_for(0.3)
    traj = plan['pre_trajectory'].joint_trajectory
    planned = dict(zip(traj.joint_names, traj.points[-1].positions))
    deviation = max(abs(planned[n] - q) for n, q in zip(ARM_JOINTS, robot.joint_positions))
    robot.get_logger().info(f'TEMP pre-grasp deviation {deviation:.4f} rad')
    robot.line_move(goals['grasp'], 'Approach')

    robot.set_contact_allowed(True)
    confirm_stage('Close gripper')
    robot.get_logger().info('Close gripper')
    if not robot.close_gripper():
        raise RuntimeError('Close gripper failed')
    log_fingers(robot, 'after close')

    confirm_stage('Lift')
    robot.get_logger().info('Lift')
    robot.line_move(goals['lift'], 'Lift')
    log_fingers(robot, 'after lift') 

    confirm_stage('Up')
    robot.get_logger().info('Up')
    robot.spin_for(0.3)
    up = robot.plan_joint_path(
        list(robot.joint_positions), robot.get_parameter('up_joints').value)
    if up is None or not robot.execute_trajectory(up):
        raise RuntimeError('Up move failed')
    log_fingers(robot, 'after up')


def approve(robot, grasp):
    """Safety gate before any arm motion. Returns True only if execution is allowed."""
    robot.get_logger().info(
        f'Planned pick at {np.round(grasp[:3, 3], 3).tolist()} in {robot.frame_id}')
    if SAFETY['dry_run']:
        robot.get_logger().info('Dry run: plan found, not executing')
        return False
    if SAFETY['confirm_execution'] and input('Execute? [y/N] ').strip().lower() != 'y':
        robot.get_logger().info('Execution declined')
        return False
    return True


def pick(robot, placed, proposal_count):
    """Best placement first; executes the first one whose whole pick is feasible."""
    for offset, rank, grasp, width, retreat in placed:
        label = 'top-down fallback' if rank == proposal_count else f'{rank}/{proposal_count}'
        robot.get_logger().info(
            f'Proposal {label}: closing centre {offset:.2f} of the half-diagonal from the '
            f'centroid, {width:.3f} m free across, backed off {retreat:.3f} m')
        c = grasp[:3, 3] + grasp[:3, :3] @ robot.hand_centre
        robot.get_logger().info(f'TEMP closing centre {np.round(c, 3).tolist()} obj centre {np.round(robot.object_centre, 3).tolist()} size {np.round(robot.object_size, 3).tolist()} approach {np.round(grasp[:3, 0], 2).tolist()}')
        ax = grasp[:3, :3] @ robot.hand_axis
        lat = np.cross(grasp[:3, 0], ax)
        d = c - robot.grasp_centre
        robot.get_logger().info(f'TEMP offset along closing {d @ ax:.3f} lateral {d @ lat:.3f}')
        plan = robot.plan_pick(grasp)
        if plan is None:
            continue
        if not approve(robot, grasp):
            return
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
        nodes = background_nodes()
        rclpy.logging.get_logger('run_pick').info(
            f'{HARDWARE_MODE} / {SCENE_SOURCE}: starting ' + ' and '.join(nodes))
        background = [start(name, params) for name, params in nodes.items()]

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