import rclpy, numpy as np
from rclpy.node import Node
from perception_msgs.msg import SceneObjectArray

rclpy.init()
n = Node('probe')
done = []
def cb(m):
    for o in m.objects:
        p = np.frombuffer(bytes(o.cloud.data), np.float32).reshape(-1, 3)
        lo, hi = p.min(0), p.max(0)
        print(o.track_id, len(p), 'centre', np.round((lo + hi) / 2, 3), 'size', np.round(hi - lo, 3))
    done.append(1)
n.create_subscription(SceneObjectArray, '/scene_objects', cb, 5)
while not done:
    rclpy.spin_once(n)