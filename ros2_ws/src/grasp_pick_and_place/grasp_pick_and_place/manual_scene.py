import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger

from grasp_pick_and_place.config import CROP_CLEARANCE
from grasp_pick_and_place.transform_utils import matrix_to_pose

SPACING = 0.004  # m between cloud points


def box_surface(size, spacing, z_start):
    """Top and four side faces of a box standing on z=0, centred on the origin in xy."""
    sx, sy, sz = size

    def axis(length, lo=None):
        lo = -length / 2.0 if lo is None else lo
        hi = lo + length
        return np.linspace(lo, hi, max(int(round((hi - lo) / spacing)) + 1, 2))

    xs, ys, zs = axis(sx), axis(sy), axis(sz - z_start, z_start)
    gx, gy = np.meshgrid(xs, ys)
    faces = [np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, sz)])]
    gy2, gz = np.meshgrid(ys, zs)
    gx2, gz2 = np.meshgrid(xs, zs)
    for sign in (-1.0, 1.0):
        faces.append(np.column_stack([np.full(gy2.size, sign * sx / 2), gy2.ravel(), gz.ravel()]))
        faces.append(np.column_stack([gx2.ravel(), np.full(gx2.size, sign * sy / 2), gz2.ravel()]))
    return np.vstack(faces)


class ManualScene(Node):
    """Scene source without a camera: a fixed box on the table, in the robot base frame."""

    def __init__(self):
        super().__init__('manual_scene')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('support_top', 0.0)
        self.declare_parameter('object_centre_xy', [0.0, 0.0])
        self.declare_parameter('object_size', [0.08, 0.08, 0.08])
        self.declare_parameter('object_yaw', 0.0)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.points_pub = self.create_publisher(PointCloud2, 'scene_points', latched)
        self.candidates_pub = self.create_publisher(PoseArray, 'grasp_candidates', latched)
        self.pose_pub = self.create_publisher(PoseStamped, 'grasp_pose', latched)
        self.create_service(Trigger, 'capture_scene', self.on_capture)

    def on_capture(self, request, response):
        frame = self.get_parameter('frame_id').value
        top = float(self.get_parameter('support_top').value)
        cx, cy = (float(v) for v in self.get_parameter('object_centre_xy').value)
        size = [float(v) for v in self.get_parameter('object_size').value]
        yaw = float(self.get_parameter('object_yaw').value)
        rot = Rotation.from_euler('z', yaw).as_matrix()
        stamp = self.get_clock().now().to_msg()

        points = box_surface(size, SPACING, CROP_CLEARANCE) @ rot.T + [cx, cy, top]
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = frame
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

        # Top-down candidate (x = approach, y = closing axis), as grasp_proposer's frame
        approach = np.array([0.0, 0.0, -1.0])
        closing = rot[:, 0]
        grasp = np.eye(4)
        grasp[:3, :3] = np.column_stack([approach, closing, np.cross(approach, closing)])
        grasp[:3, 3] = [cx, cy, top + size[2] / 2.0]

        candidates = PoseArray()
        candidates.header.stamp = stamp
        candidates.header.frame_id = frame
        candidates.poses = [matrix_to_pose(grasp)]
        pose = PoseStamped()
        pose.header = candidates.header
        pose.pose = candidates.poses[0]

        self.points_pub.publish(cloud)
        self.candidates_pub.publish(candidates)
        self.pose_pub.publish(pose)
        response.success = True
        response.message = f'{points.shape[0]} points'
        return response


def main():
    rclpy.init()
    node = ManualScene()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()