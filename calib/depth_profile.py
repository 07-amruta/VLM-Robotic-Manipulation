import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

rclpy.init()
n = Node('depth_profile')
got = []
n.create_subscription(Image, '/camera/camera/depth/image_rect_raw', got.append, qos_profile_sensor_data)
while rclpy.ok() and not got:
    rclpy.spin_once(n, timeout_sec=0.1)
m = got[0]
d = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.width) * 0.001
h, w = d.shape
strip = d[:, int(0.4 * w):int(0.6 * w)]          # centre columns
for r in range(int(0.4 * h), h, h // 24):
    band = strip[r:r + h // 24]
    v = band[band > 0]
    print(f'rows {r:3d}-{r + h // 24:3d}: median {np.median(v):.3f} m' if v.size else f'rows {r}: none')
rclpy.shutdown()