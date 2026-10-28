import sys
import glob, os

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

from grasp_pick_and_place.transform_utils import lookup_matrix

ROOT, CLOUD = 'camera_link', 'camera_depth_optical_frame'
name = sys.argv[1]                    # table, A, B or C

rclpy.init()
node = Node('calib_capture')
buf = tf2_ros.Buffer()
tf2_ros.TransformListener(buf, node)
got = []
latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
node.create_subscription(PointCloud2, 'scene_points', got.append, latched)
client = node.create_client(Trigger, 'capture_scene')
while not client.wait_for_service(timeout_sec=1.0):
    print('waiting for /capture_scene (is scene_snapshot running?)')

end = node.get_clock().now().nanoseconds + 1e9      # flush a stale latched cloud
while node.get_clock().now().nanoseconds < end:
    rclpy.spin_once(node, timeout_sec=0.05)
got.clear()

future = client.call_async(Trigger.Request())
rclpy.spin_until_future_complete(node, future)
print(future.result().message)
while rclpy.ok() and not got:
    rclpy.spin_once(node, timeout_sec=0.1)

msg = got[0]
pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3).astype(np.float64)
latest = max(glob.glob('/tmp/optical_*.npy'), key=os.path.getmtime)
print('using', latest)
np.save(f'/tmp/{name}.npy', np.load(latest).astype(np.float64))
while rclpy.ok() and not buf.can_transform(ROOT, CLOUD, rclpy.time.Time()):
    rclpy.spin_once(node, timeout_sec=0.1)
np.save('/tmp/root_T_cloud.npy', lookup_matrix(buf, ROOT, CLOUD))
print(name, len(pts), 'points in', msg.header.frame_id)
rclpy.shutdown()
