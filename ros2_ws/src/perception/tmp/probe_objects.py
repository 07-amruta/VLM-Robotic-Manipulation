import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import rclpy
from sensor_msgs_py import point_cloud2
from rclpy.node import Node
from perception_msgs.msg import SceneObjectArray

class Probe(Node):
    def __init__(self):
        super().__init__('probe_objects')
        self.create_subscription(SceneObjectArray, 'scene_objects', self.cb, 5)
    def cb(self, m):
        out = []
        for o in m.objects:
            a = point_cloud2.read_points(o.cloud, field_names=('x', 'y', 'z'), skip_nans=True)
            if a.dtype.names:
                a = np.column_stack([a[n] for n in ('x', 'y', 'z')])
            out.append((o.track_id, len(a), (a.max(0)[2].round(3) - a.min(0)[2].round(3)).tolist(), a.mean(0).round(2).tolist()))
        self.get_logger().info(f'n={len(out)} {out}')

rclpy.init()
rclpy.spin(Probe())