import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class Probe(Node):
    def __init__(self):
        super().__init__('probe_scene')
        self.create_subscription(PointCloud2, '/scene_points', self.cb, 5)

    def cb(self, m):
        a = point_cloud2.read_points(m, field_names=('x', 'y', 'z'), skip_nans=True)
        if a.dtype.names:
            a = np.column_stack([a[n] for n in ('x', 'y', 'z')])
        lo, hi = a.min(0), a.max(0)
        self.get_logger().info(
            f'frame={m.header.frame_id} n={len(a)} min={lo.round(3)} max={hi.round(3)} extent={(hi - lo).round(3)}')


rclpy.init()
rclpy.spin(Probe())