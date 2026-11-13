import cv2
import numpy as np
from cv_bridge import CvBridge
from perception_msgs.msg import InstanceMask, InstanceMaskArray
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from ultralytics import YOLOE

from perception.ros_utils import ConfiguredNode, run_node


class OpenVocabSegmenter(ConfiguredNode):
    """On-demand producer: segments the best match for a free-text noun in the colour image."""

    def __init__(self):
        super().__init__('open_vocab_segmenter')
        self.bridge = CvBridge()
        self.info = None
        self.noun = None
        self.model = YOLOE(self.param('ov_model'))
        latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(CameraInfo, self.param('color_info_topic'), self.on_info, latest)
        self.create_subscription(Image, self.param('color_topic'), self.on_image, latest)
        self.create_subscription(String, 'target_request', self.on_request, latched)
        self.pub = self.create_publisher(InstanceMaskArray, 'instance_masks/target', 10)

    def on_info(self, msg):
        self.info = msg

    def on_request(self, msg):
        noun = msg.data.strip()
        if not noun:
            return
        self.model.set_classes([noun], self.model.get_text_pe([noun]))  # once per request
        self.noun = noun
        self.get_logger().info(f'target request: {noun!r}')

    def on_image(self, msg):
        if self.noun is None or self.info is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        result = self.model.predict(
            frame, conf=self.param('ov_conf'), device=self.param('yolo_device'),
            retina_masks=True, verbose=False)[0]
        out = InstanceMaskArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.param('color_frame')
        out.camera_info = self.info
        if result.masks is not None and len(result.boxes):
            best = int(result.boxes.conf.argmax())
            mask = (result.masks.data[best].cpu().numpy() > 0.5).astype(np.uint8) * 255
            if self.param('mask_erode_px') > 0:
                mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=self.param('mask_erode_px'))
            if mask.any():
                inst = InstanceMask()
                inst.track_id = 0
                inst.class_name = self.noun
                inst.score = float(result.boxes.conf[best])
                inst.mask = self.bridge.cv2_to_imgmsg(mask, 'mono8')
                out.instances.append(inst)
                self.get_logger().info(
                    f'{self.noun!r}: score {inst.score:.2f}, mask {int((mask > 0).sum())} px',
                    throttle_duration_sec=2.0)
        if not out.instances:
            self.get_logger().info(f'{self.noun!r}: no detection', throttle_duration_sec=2.0)
        self.pub.publish(out)


def main():
    run_node(OpenVocabSegmenter)