import numpy as np
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


def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class SceneSnapshot(Node):
    """On request, deprojects the latest depth image into the target frame and crops it to the workspace."""

    def __init__(self):
        super().__init__('scene_snapshot')

        self.declare_parameter('depth_topic', '/scene_camera/depth_image')
        self.declare_parameter('camera_info_topic', '/scene_camera/camera_info')
        self.declare_parameter('optical_frame', 'scene_camera_optical')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('crop_min', [-0.4, 0.45, 0.31])
        self.declare_parameter('crop_max', [0.4, 1.05, 0.8])

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
        if depth_msg.encoding != '32FC1':
            response.success = False
            response.message = f'Unsupported depth encoding: {depth_msg.encoding}'
            return response

        target_frame = self.get_parameter('target_frame').value
        optical_frame = self.get_parameter('optical_frame').value
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, optical_frame, Time())
        except TransformException as e:
            response.success = False
            response.message = f'TF lookup failed: {e}'
            return response

        # Deproject: pixel (u, v) with depth z -> optical-frame point (x right, y down, z forward)
        depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
            depth_msg.height, depth_msg.width)
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        v, u = np.indices(depth.shape)
        valid = np.isfinite(depth) & (depth > 0.0)
        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        points = np.stack((x, y, z), axis=1)

        # Optical frame -> target frame
        t = tf.transform.translation
        q = tf.transform.rotation
        rot = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        points = points @ rot.T + np.array([t.x, t.y, t.z])

        # Crop to the workspace box
        lo = np.array(self.get_parameter('crop_min').value)
        hi = np.array(self.get_parameter('crop_max').value)
        points = points[np.all((points >= lo) & (points <= hi), axis=1)]

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