import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image

from grasp_pick_and_place.projection import deproject, depth_to_metres

PATCH = 100  # half-size of the patch used for the temporal noise

bag_path = sys.argv[1]
index = int(sys.argv[2]) if len(sys.argv) > 2 else 100
patch_row = int(sys.argv[3]) if len(sys.argv) > 3 else None
patch_col = int(sys.argv[4]) if len(sys.argv) > 4 else None

reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
            rosbag2_py.ConverterOptions('', ''))

info, chosen, encoding = None, None, None
valid_fractions, patches = [], []
while reader.has_next():
    topic, data, _ = reader.read_next()
    if topic.endswith('camera_info'):
        info = info or deserialize_message(data, CameraInfo)
        continue
    if not topic.endswith('image_rect_raw'):
        continue
    msg = deserialize_message(data, Image)
    depth = depth_to_metres(msg.data, msg.encoding, msg.height, msg.width)
    valid_fractions.append(float((np.isfinite(depth) & (depth > 0.0)).mean()))
    r = patch_row if patch_row is not None else msg.height // 2
    c = patch_col if patch_col is not None else msg.width // 2
    patches.append(depth[r - PATCH:r + PATCH, c - PATCH:c + PATCH])
    if len(valid_fractions) - 1 == index:
        chosen, encoding = depth, msg.encoding

vf = np.array(valid_fractions)
print(f'frames {len(vf)}, encoding {encoding}')
print(f'valid pixels per frame: min {vf.min():.3f} mean {vf.mean():.3f} max {vf.max():.3f}')

valid = chosen[np.isfinite(chosen) & (chosen > 0.0)]
pct = np.percentile(valid, [0, 5, 50, 95, 100])
print(f'frame {index} depth m: min {pct[0]:.3f} p5 {pct[1]:.3f} median {pct[2]:.3f} '
      f'p95 {pct[3]:.3f} max {pct[4]:.3f}')

stack = np.stack(patches)
stable = (stack > 0.0).all(axis=0)
noise_mm = np.median(stack.std(axis=0)[stable]) * 1000.0
print(f'temporal noise, central patch: median per-pixel std {noise_mm:.2f} mm '
      f'({stable.mean():.2f} of the patch valid in every frame)')

points, _ = deproject(chosen, np.array(info.k).reshape(3, 3))
print(f'{len(points)} points; x [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] '
      f'y [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] '
      f'z [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]')

plt.imshow(np.where(chosen > 0.0, chosen, np.nan), cmap='turbo', vmin=pct[1], vmax=pct[3])
plt.colorbar(label='depth (m)')
out = bag_path.rstrip('/') + '_depth.png'
plt.savefig(out, dpi=100)
print(f'saved {out}')