import numpy as np
import rclpy
import tf2_ros

from grasp_pick_and_place.config import GRIPPER, ROBOT
from grasp_pick_and_place.transform_utils import lookup_matrix

TCP = ROBOT['tcp_frame']
rclpy.init()
node = rclpy.create_node('hand_centre')
buf = tf2_ros.Buffer()
tf2_ros.TransformListener(buf, node)
frames = [TCP, 'base_link', GRIPPER['hand_opposing_frame'], *GRIPPER['hand_pair_frames']]
while rclpy.ok() and not all(buf.can_transform(TCP, f, rclpy.time.Time()) for f in frames):
    rclpy.spin_once(node, timeout_sec=0.1)
end = node.get_clock().now().nanoseconds + 5e8
while node.get_clock().now().nanoseconds < end:
    rclpy.spin_once(node, timeout_sec=0.05)


def pos(frame):
    return lookup_matrix(buf, TCP, frame)[:3, 3]


pair = np.mean([pos(f) for f in GRIPPER['hand_pair_frames']], axis=0)
centre = (pair + pos(GRIPPER['hand_opposing_frame'])) / 2.0
centre[0] = 0.0                                     # x is the approach axis
base_T_tcp = lookup_matrix(buf, 'base_link', TCP)
print('closing centre in base_link', np.round((base_T_tcp @ [*centre, 1.0])[:3], 4).tolist())
print('approach axis in base_link', np.round(base_T_tcp[:3, 0], 3).tolist(),
      '(want x,y components under 0.03)')
rclpy.shutdown()