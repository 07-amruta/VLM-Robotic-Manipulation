import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
                          
import time
from collections import deque

import numpy as np
import cv2
cv2.setNumThreads(1)

from cv_bridge import CvBridge
from perception_msgs.msg import InstanceMaskArray, SceneObject, SceneObjectArray
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from perception.mask_to_cloud import object_clouds, to_pointcloud2
from perception.projection import clean_depth, deproject, depth_to_metres
from perception.ros_utils import ConfiguredNode, lookup_matrix, run_node


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


class SceneObjects(ConfiguredNode):
    """Consumer: instance masks + nearest-stamp depth -> per-object clouds in the target frame."""

    def __init__(self):
        super().__init__('scene_objects')
        self.bridge = CvBridge()
        self.depth_buffer = deque(maxlen=10)
        self.info = None
        self.last_run = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, self.param('depth_topic'), self.depth_buffer.append, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.param('camera_info_topic'), self.on_info, qos_profile_sensor_data)
        self.create_subscription(InstanceMaskArray, self.param('masks_topic'), self.on_masks, 5)
        self.pub = self.create_publisher(SceneObjectArray, 'scene_objects', 5)

    def on_info(self, msg):
        self.info = msg

    def on_masks(self, array):
        now = time.monotonic()
        if not self.depth_buffer or self.info is None or now - self.last_run < 1.0 / self.param('objects_rate_hz'):
            return
        target_frame, depth_frame = self.param('target_frame'), self.param('optical_frame')
        try:
            to_target = lookup_matrix(self.tf_buffer, target_frame, depth_frame)
            to_mask = lookup_matrix(self.tf_buffer, array.header.frame_id, depth_frame)
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return
        self.last_run = now

        t = stamp_ns(array)
        msg = min(self.depth_buffer, key=lambda m: abs(stamp_ns(m) - t))
        depth = clean_depth(depth_to_metres(msg.data, msg.encoding, msg.height, msg.width),
                            self.param('depth_min'), self.param('depth_max'), self.param('edge_tol'))
        points, _ = deproject(depth, np.array(self.info.k).reshape(3, 3))
        instances = [(i.track_id, i.class_name, i.score, self.bridge.imgmsg_to_cv2(i.mask, 'mono8') > 0)
                     for i in array.instances]
        found = object_clouds(
            points, to_mask, to_target, np.array(array.camera_info.k).reshape(3, 3), instances,
            self.param('outlier_k'), self.param('outlier_std'), self.param('min_object_points'))

        out = SceneObjectArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = target_frame
        for track_id, name, score, cloud in found:
            obj = SceneObject()
            obj.track_id, obj.class_name, obj.score = int(track_id), name, float(score)
            obj.cloud = to_pointcloud2(cloud, out.header)
            out.objects.append(obj)
        self.pub.publish(out)


def main():
    run_node(SceneObjects)