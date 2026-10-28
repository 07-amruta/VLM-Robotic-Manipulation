import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

rng = np.random.default_rng(0)
THR = 0.006     # m, RANSAC inlier distance


def fit_plane(p, iters=400):
    best = None
    for _ in range(iters):
        a = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(a[1] - a[0], a[2] - a[0]); k = np.linalg.norm(n)
        if k < 1e-9: continue
        n /= k
        inl = np.abs((p - a[0]) @ n) < THR
        if best is None or inl.sum() > best.sum(): best = inl
    q = p[best]; c = q.mean(0)
    n = np.linalg.svd(q - c)[2][-1]
    n = -n if n @ c > 0 else n                    # normal faces the camera
    return n, float(-(n @ c)), best


rclpy.init()
node = Node('plane_check')
img, info = [], []
node.create_subscription(Image, '/camera/camera/depth/image_rect_raw', img.append, qos_profile_sensor_data)
node.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', info.append, qos_profile_sensor_data)
while rclpy.ok() and not (img and info):
    rclpy.spin_once(node, timeout_sec=0.1)
m, k = img[0], info[0].k
d = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.width) * 0.001
v, u = np.nonzero((d > 0.4) & (d < 2.0))
z = d[v, u]
pts = np.column_stack([(u - k[2]) * z / k[0], (v - k[5]) * z / k[4], z])
pts = pts[rng.choice(len(pts), min(len(pts), 60000), replace=False)]

planes = []
for i in range(2):
    n, off, inl = fit_plane(pts)
    rms = np.std(pts[inl] @ n + off) * 1000
    tilt = 90 - np.degrees(np.arccos(abs(n[2])))
    print(f'plane {i + 1}: normal {np.round(n, 3).tolist()}  distance {off:.3f} m  '
          f'inliers {inl.sum()}  RMS {rms:.1f} mm  mean depth {pts[inl][:, 2].mean():.2f} m  '
          f'optical axis {tilt:.0f} deg below horizontal')
    planes.append((n, off))
    pts = pts[~inl]
if len(planes) == 2:
    cosang = float(np.clip(planes[0][0] @ planes[1][0], -1, 1))
    print('angle between planes', round(float(np.degrees(np.arccos(cosang))), 1), 'deg,',
          'offset difference', round(abs(planes[0][1] - planes[1][1]) * 1000, 1), 'mm')
rclpy.shutdown()