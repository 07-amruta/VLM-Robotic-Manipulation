import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import cv2

cv2.setNumThreads(1)

from cv_bridge import CvBridge
from perception_msgs.msg import InstanceMask, InstanceMaskArray
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener, ExtrapolationException
from rclpy.time import Time

from perception.projection import clean_depth, deproject, depth_to_metres, transform_points
from perception.ros_utils import ConfiguredNode, lookup_matrix, run_node
from perception.support_planes import cluster_labels, peel_support_planes
from perception.self_filter import link_boxes, self_keep_mask

class GeometricSegmenter(ConfiguredNode):
    """Depth-only producer: peels support planes, clusters the rest, one instance mask per cluster."""

    def __init__(self):
        super().__init__('geometric_segmenter')
        self.bridge = CvBridge()
        self.depth_msg = None
        self.info = None
        self.last_stamp = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, self.param('depth_topic'), self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.param('camera_info_topic'), self.on_info, qos_profile_sensor_data)
        self.pub = self.create_publisher(InstanceMaskArray, 'instance_masks/geometric', 10)
        self.create_timer(1.0 / self.param('rate_hz'), self.tick)
        self.boxes = None
        self.create_subscription(
            String, '/robot_description', self.on_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def on_depth(self, msg):
        self.depth_msg = msg

    def on_info(self, msg):
        self.info = msg

    def on_description(self, msg):
        self.boxes = link_boxes(msg.data)
        self.get_logger().info(f'self-filter: {len(self.boxes)} link boxes')

    def tick(self):
        msg, info = self.depth_msg, self.info
        if self.boxes is None:
            self.get_logger().warn('waiting for /robot_description', throttle_duration_sec=5.0)
            return
        if msg is None or info is None or msg.header.stamp == self.last_stamp:
            return
        frame = self.param('optical_frame')
        try:
            to_target = lookup_matrix(self.tf_buffer, self.param('target_frame'), frame)
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return
        self.last_stamp = msg.header.stamp

        depth = clean_depth(depth_to_metres(msg.data, msg.encoding, msg.height, msg.width),
                            self.param('depth_min'), self.param('depth_max'), self.param('edge_tol'))
        points, pixels = deproject(depth, np.array(info.k).reshape(3, 3))
        points = transform_points(points, to_target)
        lo, hi = np.array(self.param('reach_min')), np.array(self.param('reach_max'))
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        points, pixels = points[inside], pixels[inside]

        failed = []

        def link_to_target(name):
            target = self.param('target_frame')
            try:
                return lookup_matrix(self.tf_buffer, target, name, msg.header.stamp)
            except ExtrapolationException as e:
                lag = (Time.from_msg(msg.header.stamp).nanoseconds
                       - self.tf_buffer.get_latest_common_time(target, name).nanoseconds) * 1e-9
                if lag <= self.param('tf_tolerance_sec'):
                    return lookup_matrix(self.tf_buffer, target, name)
                failed.append(f'{name}: lag {lag:.3f}s')
                return None
            except TransformException as e:
                failed.append(f'{name}: {e}')
                return None

        keep = self_keep_mask(points, self.boxes, link_to_target, self.param('self_filter_margin'))
        if failed:
            self.get_logger().warn(
                f'self-filter: {len(failed)} link TF lookups failed (e.g. {failed[0]}); frame dropped',
                throttle_duration_sec=5.0)
            return
        points, pixels = points[keep], pixels[keep]

        out = InstanceMaskArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = frame
        out.camera_info = info
        if len(points) >= 3:
            planes, remaining = peel_support_planes(
                points, self.param('plane_tol'), self.param('plane_margin'), self.param('plane_iters'),
                self.param('plane_min_area'), self.param('max_planes'), self.param('max_tilt_deg'))
            self.get_logger().info(
                f'support planes z={[round(p["z"], 3) for p in planes]}', throttle_duration_sec=5.0)
            rest, rest_pixels = points[remaining], pixels[remaining]
            if len(rest):
                labels = cluster_labels(rest, self.param('cluster_radius'), self.param('cluster_voxel'))
                counts = np.bincount(labels)
                keep = [l for l in np.argsort(-counts) if counts[l] >= self.param('min_cluster_points')]
                for track_id, label in enumerate(keep[:self.param('max_instances')]):
                    sel = rest_pixels[labels == label]
                    mask = np.zeros((msg.height, msg.width), np.uint8)
                    mask[sel[:, 1], sel[:, 0]] = 255
                    inst = InstanceMask()
                    inst.track_id = track_id  # per-frame index, largest first; no tracking yet
                    inst.class_name = 'object'
                    inst.score = 1.0
                    inst.mask = self.bridge.cv2_to_imgmsg(mask, 'mono8')
                    out.instances.append(inst)
        self.pub.publish(out)


def main():
    run_node(GeometricSegmenter)