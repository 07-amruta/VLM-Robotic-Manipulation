import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

class CollisionChecker(Node):
    def __init__(self):
        super().__init__('collision_checker')
        self.cli = self.create_client(GetStateValidity, '/check_state_validity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for /check_state_validity...')

    def check(self, joint_names, positions):
        req = GetStateValidity.Request()
        req.robot_state = RobotState()
        req.robot_state.joint_state = JointState()
        req.robot_state.joint_state.name = joint_names
        req.robot_state.joint_state.position = positions
        req.group_name = 'gripper'
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.get_logger().info(f'valid: {res.valid}')
        if not res.valid:
            self.get_logger().warning('State is invalid; collisions follow:')
            for c in res.contacts:
                self.get_logger().warning(f'collision: {c.contact_body_1} <-> {c.contact_body_2}')
        else:
            self.get_logger().info('No collision reported for this state.')

def main():
    rclpy.init()
    node = CollisionChecker()
    node.check(
        ['gripper_finger_1_joint_1', 'gripper_finger_2_joint_1', 'gripper_finger_middle_joint_1'],
        [1.0, 1.0, 1.0]
    )
    rclpy.shutdown()

if __name__ == '__main__':
    main()