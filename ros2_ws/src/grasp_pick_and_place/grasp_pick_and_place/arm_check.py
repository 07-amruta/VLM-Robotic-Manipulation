import sys

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from grasp_pick_and_place.config import ARM_JOINTS

ACTION = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
JOINT = 'wrist_3_joint'
DELTA = 0.1      # rad
DURATION = 5     # s


def go(node, client, positions):
    point = JointTrajectoryPoint(
        positions=[float(p) for p in positions], time_from_start=Duration(sec=DURATION))
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ARM_JOINTS
    goal.trajectory.points = [point]
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if not handle.accepted:
        return False
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    return result_future.result().result.error_code == 0


def main():
    rclpy.init()
    node = Node('arm_check')
    log = node.get_logger()
    state = {}
    node.create_subscription(
        JointState, '/joint_states', lambda m: state.update(zip(m.name, m.position)), 10)
    while rclpy.ok() and not all(j in state for j in ARM_JOINTS):
        rclpy.spin_once(node, timeout_sec=0.1)
    start = [state[j] for j in ARM_JOINTS]
    log.info('joints: ' + str([round(q, 3) for q in start]))

    if '--move' in sys.argv or '--dry' in sys.argv:
        client = ActionClient(node, FollowJointTrajectory, ACTION)
        if not client.wait_for_server(timeout_sec=5.0):
            log.error(f'{ACTION} not available')
        else:
            index = ARM_JOINTS.index(JOINT)
            target = list(start)
            target[index] += DELTA
            if '--dry' in sys.argv:
                log.info(f'dry run, nothing sent: {JOINT} {start[index]:.3f} -> '
                         f'{target[index]:.3f} rad over {DURATION} s')
            else:
                input(f'Clear the workspace, e-stop in reach. Enter to move {JOINT} by {DELTA} rad ')
                log.info(f'out: {go(node, client, target)}')
                log.info(f'back: {go(node, client, start)}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()