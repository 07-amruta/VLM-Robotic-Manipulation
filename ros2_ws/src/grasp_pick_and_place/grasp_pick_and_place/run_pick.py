import signal
import subprocess
import time
import os

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_srvs.srv import Trigger

PKG = 'grasp_pick_and_place'
CAPTURE_SRV = '/capture_scene'
POSE_TOPIC = '/grasp_pose'
READY_TIMEOUT = 120.0
POSE_TIMEOUT = 20.0  # per capture attempt
CAPTURE_ATTEMPTS = 3


def start(executable):
    return subprocess.Popen(
        ['ros2', 'run', PKG, executable], start_new_session=True)


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


def main():
    rclpy.init()
    node = Node('run_pick')
    background = []
    try:
        node.get_logger().info('Starting scene_snapshot and grasp_proposer')
        background.append(start('scene_snapshot'))
        background.append(start('grasp_proposer'))

        client = node.create_client(Trigger, CAPTURE_SRV)
        if not spin_until(node, client.service_is_ready, READY_TIMEOUT):
            raise RuntimeError(f'{CAPTURE_SRV} not available')
        if not spin_until(
                node, lambda: node.count_publishers(POSE_TOPIC) > 0, READY_TIMEOUT):
            raise RuntimeError(f'{POSE_TOPIC} has no publisher')

        received = []
        node.create_subscription(PoseStamped, POSE_TOPIC, received.append, 10)
        spin_until(node, lambda: False, 2.0)  # let the subscription connect

        for attempt in range(1, CAPTURE_ATTEMPTS + 1):
            node.get_logger().info(
                f'Capturing scene (attempt {attempt}/{CAPTURE_ATTEMPTS})')
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future)
            if spin_until(node, lambda: bool(received), POSE_TIMEOUT):
                break
        else:
            raise RuntimeError('No grasp pose received')
        pose = received[0]

        node.get_logger().info(
            f'Best grasp in {pose.header.frame_id}: '
            f'position=({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, '
            f'{pose.pose.position.z:.3f}) '
            f'orientation=({pose.pose.orientation.x:.3f}, {pose.pose.orientation.y:.3f}, '
            f'{pose.pose.orientation.z:.3f}, {pose.pose.orientation.w:.3f})')

        node.get_logger().info('Running pick_executor')
        subprocess.run(['ros2', 'run', PKG, 'pick_executor'])
    except RuntimeError as err:
        node.get_logger().error(str(err))
    finally:
        for proc in background:
            stop(proc)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()