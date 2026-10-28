import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

rclpy.init()
n = Node('depth_check')
got = []
n.create_subscription(Image, '/camera/camera/depth/image_rect_raw', got.append, qos_profile_sensor_data)
while rclpy.ok() and not got:
    rclpy.spin_once(n, timeout_sec=0.1)
m = got[0]
print('encoding', m.encoding, m.width, 'x', m.height)
d = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.width) * 0.001
print('valid fraction', round(float((d > 0).mean()), 3))
h, w = d.shape
for r in range(3):
    row = []
    for c in range(3):
        p = d[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
        v = p[p > 0]
        row.append(f'{np.median(v):.2f} m ({v.size / p.size:.0%})' if v.size else 'none')
    print(row)
rclpy.shutdown()