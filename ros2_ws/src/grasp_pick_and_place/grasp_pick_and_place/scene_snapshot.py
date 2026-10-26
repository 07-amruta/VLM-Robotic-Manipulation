import numpy as np
import time as _t
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from grasp_pick_and_place.projection import deproject, depth_to_metres, transform_points
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import maximum_filter, minimum_filter


def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

def fit_plane(points, iters, tol):
    """RANSAC dominant plane; returns unit normal (z >= 0) and offset d, with n.p + d = 0."""
    rng = np.random.default_rng()
    best = None
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        d = -n @ s[0]
        count = (np.abs(points @ n + d) < tol).sum()
        if best is None or count > best[0]:
            best = (count, n, d)
    return best[1], best[2]

def largest_cluster(points, radius):
    """Largest connected component, linking points closer than `radius`."""
    pairs = cKDTree(points).query_pairs(radius, output_type='ndarray')
    n = len(points)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return points[labels == np.bincount(labels).argmax()]

class SceneSnapshot(Node):
    """On request, deprojects the latest depth image into the target frame and crops it to the workspace."""

    def __init__(self):
        super().__init__('scene_snapshot')

        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('optical_frame', 'camera_depth_optical_frame')
        self.declare_parameter('depth_min', 0.0)
        self.declare_parameter('depth_max', 100.0)
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('crop_min', [-0.335, 0.518, 0.445])
        self.declare_parameter('crop_max', [-0.195, 0.658, 0.586])
        self.declare_parameter('plane_tol', 0.008)
        self.declare_parameter('plane_margin', 0.015)
        self.declare_parameter('plane_iters', 200)
        self.declare_parameter('cluster_radius', 0.015)
        self.declare_parameter('edge_tol', 0.01)
        self.declare_parameter('support_z', float('nan'))  # nan = off (sim unchanged)

        self.depth_msg = None
        self.info_msg = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Image, self.get_parameter('depth_topic').value,
            self.on_depth, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.on_info, qos_profile_sensor_data)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.points_pub = self.create_publisher(PointCloud2, 'scene_points', latched)
        self.create_service(Trigger, 'capture_scene', self.on_capture)

    def on_depth(self, msg):
        self.depth_msg = msg

    def on_info(self, msg):
        self.info_msg = msg

    def on_capture(self, request, response):
        if self.depth_msg is None or self.info_msg is None:
            response.success = False
            response.message = 'No depth image or camera_info received yet'
            return response

        depth_msg, info = self.depth_msg, self.info_msg
        try:
            depth = depth_to_metres(
                depth_msg.data, depth_msg.encoding, depth_msg.height, depth_msg.width)
        except ValueError as e:
            response.success = False
            response.message = str(e)
            return response

        target_frame = self.get_parameter('target_frame').value
        optical_frame = self.get_parameter('optical_frame').value
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, optical_frame, Time())
        except TransformException as e:
            response.success = False
            response.message = f'TF lookup failed: {e}'
            return response

        # Drop returns outside the camera's valid range (e.g. sentinel values)
        depth_lo, depth_hi = self.get_parameter('depth_min').value, self.get_parameter('depth_max').value
        depth = np.where((depth >= depth_lo) & (depth <= depth_hi), depth, 0.0)

        # Drop depth-edge pixels (stereo edge smear): large depth jump inside the 3x3 neighbourhood
        valid = depth > 0
        hi = maximum_filter(depth, size=3)
        lo = minimum_filter(np.where(valid, depth, np.inf), size=3)
        depth = np.where(valid & (hi - lo <= self.get_parameter('edge_tol').value), depth, 0.0)

        # Deproject to the optical frame (x right, y down, z forward), then to the target frame
        points, _ = deproject(depth, np.array(info.k).reshape(3, 3))
        np.save(f'/tmp/optical_{_t.time_ns()}.npy', points.astype(np.float32))  # temporary
        t = tf.transform.translation
        q = tf.transform.rotation
        matrix = np.eye(4)
        matrix[:3, :3] = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        matrix[:3, 3] = [t.x, t.y, t.z]
        points = transform_points(points, matrix)        

        # Crop to the workspace box
        lo = np.array(self.get_parameter('crop_min').value)
        hi = np.array(self.get_parameter('crop_max').value)
        points = points[np.all((points >= lo) & (points <= hi), axis=1)]
        if points.shape[0] < 3:
            response.success = False
            response.message = 'Too few points in crop'
            return response
        support_z = self.get_parameter('support_z').value
        if np.isnan(support_z):
            n, d = fit_plane(points, self.get_parameter('plane_iters').value,
                             self.get_parameter('plane_tol').value)
            tol = self.get_parameter('plane_tol').value
            plane_z = points[np.abs(points @ n + d) < tol][:, 2].mean()
            self.get_logger().info(f'plane n={np.round(n, 3).tolist()} z={plane_z:.3f}')
        else:  # calibrated, levelled frame: the table is the plane z = support_z
            n, d = np.array([0.0, 0.0, 1.0]), -float(support_z)
        points = points[points @ n + d > self.get_parameter('plane_margin').value]
        if points.shape[0] == 0:
            response.success = False
            response.message = 'No points above support plane'
            return response
        points = largest_cluster(points, self.get_parameter('cluster_radius').value)

        cloud = PointCloud2()
        cloud.header.stamp = depth_msg.header.stamp
        cloud.header.frame_id = target_frame
        cloud.height = 1
        cloud.width = points.shape[0]
        cloud.fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate('xyz')
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * cloud.width
        cloud.is_dense = True
        cloud.data = points.astype(np.float32).tobytes()
        self.points_pub.publish(cloud)

        response.success = points.shape[0] > 0
        response.message = f'{points.shape[0]} points in crop'
        return response


def main():
    rclpy.init()
    node = SceneSnapshot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise  # a real failure; only swallow errors raised while the context shuts down
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()